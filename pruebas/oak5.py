import depthai as dai
import numpy as np
import cv2
from datetime import datetime
import os
import logging
from logging.handlers import RotatingFileHandler
import open3d as o3d

DOT_PROJECTOR_INTENSITY = 1
FLOOD_LIGHT_INTENSITY = 0

# Crear el directorio de logs si no existe
log_directory = os.path.join(os.getcwd(), 'logs')
os.makedirs(log_directory, exist_ok=True)

# Nombre del archivo de log con fecha
log_filename = os.path.join(log_directory, f'silo_register_{datetime.now().strftime("%Y%m%d")}.log')

# Configurar el logger
logger = logging.getLogger('silo_register')
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    # Formato del log
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Handler para archivo con rotación (máximo 5MB por archivo, máximo 10 archivos)
    file_handler = RotatingFileHandler(
        log_filename,
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=10,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Handler para la terminal
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Agregar los handlers al logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# Configuración
logger.info('Importando variables...')
node = int(os.getenv('NODE', 0))
data_directory = os.getenv('DATA_DIRECTORY_EXPORT', '/home/gerson/git/silo_cam/labs')
image_infra = os.getenv('IMAGE_IR', 'False').lower() == 'true'

FPS = 5
FRAME_TARGET = 50

logger.info("Inicializando pipeline y nodos de cámara...")

# Crear pipeline
pipeline = dai.Pipeline()

# Crear nodos
mono_left = pipeline.create(dai.node.MonoCamera)
mono_right = pipeline.create(dai.node.MonoCamera)
depth = pipeline.create(dai.node.StereoDepth)
pointcloud = pipeline.create(dai.node.PointCloud)

xout_pointcloud = pipeline.create(dai.node.XLinkOut)
xout_pointcloud.setStreamName("pcl")

xout_mono_right = pipeline.create(dai.node.XLinkOut)
xout_mono_right.setStreamName("mono_right")

logger.info("Configurando cámaras mono...")

# Configuración de cámaras
mono_resolution = dai.MonoCameraProperties.SensorResolution.THE_480_P
mono_left.setResolution(mono_resolution)
mono_left.setCamera("left")
mono_left.setFps(FPS)

mono_right.setResolution(mono_resolution)
mono_right.setCamera("right")
mono_right.setFps(FPS)

logger.info("Configurando módulo de profundidad...")

depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_5x5)
depth.setLeftRightCheck(True)
depth.setExtendedDisparity(False)
depth.setSubpixel(True)

resolution_dimensions = {
    dai.MonoCameraProperties.SensorResolution.THE_400_P: (640, 400),
    dai.MonoCameraProperties.SensorResolution.THE_480_P: (640, 480),
    dai.MonoCameraProperties.SensorResolution.THE_720_P: (1280, 720),
    dai.MonoCameraProperties.SensorResolution.THE_800_P: (1280, 800),
}

if mono_resolution in resolution_dimensions:
    width, height = resolution_dimensions[mono_resolution]
    depth.setOutputSize(width, height)

config = depth.initialConfig.get()
config.costMatching.confidenceThreshold = 220
config.postProcessing.spatialFilter.enable = True
config.postProcessing.spatialFilter.holeFillingRadius = 2
config.postProcessing.spatialFilter.numIterations = 2
config.postProcessing.temporalFilter.enable = True
config.postProcessing.temporalFilter.alpha = 0.4
config.postProcessing.temporalFilter.delta = 30
config.postProcessing.speckleFilter.enable = True
config.postProcessing.speckleFilter.speckleRange = 60
config.postProcessing.thresholdFilter.minRange = 300
config.postProcessing.thresholdFilter.maxRange = 10000
depth.initialConfig.set(config)

depth.setDepthAlign(dai.CameraBoardSocket.LEFT)

logger.info("Enlazando nodos del pipeline...")

mono_left.out.link(depth.left)
mono_right.out.link(depth.right)
depth.depth.link(pointcloud.inputDepth)
pointcloud.outputPointCloud.link(xout_pointcloud.input)
mono_right.out.link(xout_mono_right.input)

logger.info("Pipeline configurado correctamente.")

try:
    with dai.Device(pipeline) as device:
        logger.info("Dispositivo conectado y pipeline cargado.")

        q_pcl = device.getOutputQueue(name="pcl", maxSize=1, blocking=True)
        q_mono_right = device.getOutputQueue(name="mono_right", maxSize=1, blocking=True)

        logger.info("Configurando proyector IR...")
        device.setIrLaserDotProjectorIntensity(DOT_PROJECTOR_INTENSITY)
        device.setIrFloodLightIntensity(FLOOD_LIGHT_INTENSITY)

        logger.info("Esperando %s frames para estabilizar...", FRAME_TARGET)
        for i in range(FRAME_TARGET):
            try:
                q_pcl.get()
                q_mono_right.get()
            except Exception as exc:
                logger.warning("Error capturando frame %s: %s", i + 1, exc)
                continue

        logger.info("Capturando nube de puntos e imagen infrarroja...")

        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        filename = f'{node}_{timestamp}'

        try:
            in_pcl = q_pcl.get()
            points = in_pcl.getPoints().astype(np.float64) / 1000.0
            points[:, 2] *= -1
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            logger.info("Nube de puntos capturada con %s puntos", points.shape[0])

            ply_filename = f"{filename}.ply"
            if o3d.io.write_point_cloud(f'{data_directory}/{ply_filename}', pcd, write_ascii=False):
                logger.info("[PLY] Guardado como: %s", ply_filename)
            else:
                logger.error("No se pudo guardar el archivo PLY: %s", ply_filename)
        except Exception:
            logger.exception("Fallo al capturar o guardar la nube de puntos")

        try:
            mono_frame = q_mono_right.get().getCvFrame()
            ir_filename = f"{filename}_ir.jpg"
            if cv2.imwrite(f'{data_directory}/{ir_filename}', mono_frame):
                logger.info("[Imagen IR] Guardada como: %s", ir_filename)
            else:
                logger.error("No se pudo guardar la imagen IR: %s", ir_filename)
        except Exception:
            logger.exception("Fallo al capturar o guardar la imagen infrarroja")


except Exception:
    logger.critical("No se pudo iniciar el dispositivo o el pipeline", exc_info=True)

logger.info("Ejecucción finalizada.")
logger.info("")
