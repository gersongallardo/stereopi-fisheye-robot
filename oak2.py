import depthai as dai
import numpy as np
import cv2
import time
from datetime import datetime
import os
import logging
from logging.handlers import RotatingFileHandler
import open3d as o3d

dot_projector_intensity=1
flood_light_intensity=0

# Crear el directorio de logs si no existe
log_directory = os.path.join(os.getcwd(), 'logs')
os.makedirs(log_directory, exist_ok=True)

# Nombre del archivo de log con fecha
log_filename = os.path.join(log_directory, f'silo_register_{datetime.now().strftime("%Y%m%d")}.log')

# Configurar el logger
logger = logging.getLogger('silo_register')
logger.setLevel(logging.DEBUG)

# Formato del log
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Handler para archivo con rotación (máximo 5MB por archivo, máximo 10 archivos)
file_handler = RotatingFileHandler(
    log_filename,
    maxBytes=5*1024*1024,  # 5MB
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
#data_directory = os.getenv('DATA_DIRECTORY_EXPORT', '/home/innovex/Documentos/oak_1m')
image_infra = os.getenv('IMAGE_IR', 'False').lower() == 'true'


FPS = 5
FRAME_TARGET = 50

logger.info("Inicializando pipeline y nodos de cámara...")

# Crear pipeline
pipeline = dai.Pipeline()

# Crear nodos
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
depth = pipeline.create(dai.node.StereoDepth)
pointcloud = pipeline.create(dai.node.PointCloud)

xoutPointCloud = pipeline.create(dai.node.XLinkOut)
xoutPointCloud.setStreamName("pcl")

#xoutMonoLeft = pipeline.create(dai.node.XLinkOut)
#xoutMonoLeft.setStreamName("mono_left")
xoutMonoRight = pipeline.create(dai.node.XLinkOut)
xoutMonoRight.setStreamName("mono_right")


logger.info("Configurando cámaras mono...")

# Configuración de cámaras
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setCamera("left")
monoLeft.setFps(FPS)

monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setCamera("right")
monoRight.setFps(FPS)

logger.info("Configurando módulo de profundidad...")

# Configuración del módulo de profundidad
depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
depth.setLeftRightCheck(True)
depth.setExtendedDisparity(False)
depth.setSubpixel(True)

logger.info("Enlazando nodos del pipeline...")

# Enlaces
monoLeft.out.link(depth.left)
monoRight.out.link(depth.right)
depth.depth.link(pointcloud.inputDepth)
pointcloud.outputPointCloud.link(xoutPointCloud.input)
#monoLeft.out.link(xoutMonoLeft.input)
monoRight.out.link(xoutMonoRight.input)

logger.info("Pipeline configurado correctamente.")

# Ejecución del dispositivo
try:
    with dai.Device(pipeline) as device:
        logger.info("Dispositivo conectado y pipeline cargado.")

        qPcl = device.getOutputQueue(name="pcl", maxSize=1, blocking=True)
        #qMonoLeft = device.getOutputQueue(name="mono_left", maxSize=1, blocking=True)
        qMonoRight = device.getOutputQueue(name="mono_right", maxSize=1, blocking=True)

        logger.info("Configurando proyector IR...")
        device.setIrLaserDotProjectorIntensity(dot_projector_intensity)
        device.setIrFloodLightIntensity(flood_light_intensity)

        logger.info("Esperando {} frames para estabilizar...".format(FRAME_TARGET))
        for i in range(FRAME_TARGET):
            try:
                qPcl.get()
                #qMonoLeft.get()
                qMonoRight.get()
            except Exception as e:
                logger.warning(f"Error capturando frame {i+1}: {e}")
                continue

        logger.info("Capturando nube de puntos e imagen infrarroja...")

        # Captura de nube de puntos
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")  # Obtener hora de muestra
        filename = f'{node}_{timestamp}'
        try:
            inPcl = qPcl.get()
            points = inPcl.getPoints().astype(np.float64)/1000.0
            points[:, 2] *= -1
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            ply_filename = f"{filename}.ply"
            if o3d.io.write_point_cloud(f'{data_directory}/{ply_filename}', pcd, write_ascii=False):
                logger.info(f"[PLY] Guardado como: {ply_filename}")
            else:
                logger.error(f"No se pudo guardar el archivo PLY: {ply_filename}")
        except Exception as e:
            logger.exception("Fallo al capturar o guardar la nube de puntos")

        # Captura de imagen infrarroja
        try:
            #monoFrame = qMonoLeft.get().getCvFrame()
            monoFrame = qMonoRight.get().getCvFrame()
            ir_filename = f"{filename}_ir.jpg"
            if cv2.imwrite(f'{data_directory}/{ir_filename}', monoFrame):
                logger.info(f"[Imagen IR] Guardada como: {ir_filename}")
            else:
                logger.error(f"No se pudo guardar la imagen IR: {ir_filename}")
        except Exception as e:
            logger.exception("Fallo al capturar o guardar la imagen infrarroja")

except Exception as e:
    logger.critical("No se pudo iniciar el dispositivo o el pipeline", exc_info=True)

logger.info(f"Ejecucción finalizada.")
logger.info("")
