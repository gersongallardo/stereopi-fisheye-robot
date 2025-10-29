import depthai as dai
import numpy as np
import cv2
from datetime import datetime
import os
import logging
from logging.handlers import RotatingFileHandler
import open3d as o3d

dot_projector_intensity = 1
flood_light_intensity = 0
FPS = 5
FRAME_TARGET = 50

# Crear directorio de logs
log_directory = os.path.join(os.getcwd(), 'logs')
os.makedirs(log_directory, exist_ok=True)
log_filename = os.path.join(log_directory, f'silo_register_{datetime.now().strftime("%Y%m%d")}.log')

# Configuración de logger
logger = logging.getLogger('silo_register')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

file_handler = RotatingFileHandler(log_filename, maxBytes=5*1024*1024, backupCount=10, encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Configuración de entorno
logger.info('Importando variables...')
node = int(os.getenv('NODE', 0))
data_directory = os.getenv('DATA_DIRECTORY_EXPORT', '/home/pi/Documents')
image_infra = os.getenv('IMAGE_IR', 'False').lower() == 'true'

logger.info("Inicializando pipeline...")

pipeline = dai.Pipeline()

# Cámaras mono
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setCamera("left")
monoLeft.setFps(FPS)

monoRight = pipeline.create(dai.node.Camera)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
monoRight.setResolution(dai.ColorCameraProperties.SensorResolution.THE_400_P)
monoRight.setFps(FPS)

# Crear output queue al host
monoRightOut = monoRight.output


# StereoDepth
depth = pipeline.create(dai.node.StereoDepth)
depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
depth.setLeftRightCheck(True)
depth.setExtendedDisparity(False)
depth.setSubpixel(True)

# Streams
depth.setOutputDepth(True)
depth.setOutputRectified(True)
depth.setStreamName("depth")

# Enlaces
monoLeft.out.link(depth.left)

logger.info("Pipeline configurado correctamente.")

try:
    with dai.Device(pipeline) as device:
        logger.info("Dispositivo conectado y pipeline cargado.")

        qDepth = device.getOutputQueue("depth", maxSize=1, blocking=True)
        qMonoRight = device.getOutputQueue("mono_right", maxSize=1, blocking=True)  

        logger.info("Configurando proyector IR...")
        device.setIrLaserDotProjectorIntensity(dot_projector_intensity)
        device.setIrFloodLightIntensity(flood_light_intensity)

        logger.info(f"Esperando {FRAME_TARGET} frames para estabilizar...")
        for i in range(FRAME_TARGET):
            try:
                qDepth.get()
                qMonoRight.get()
            except Exception as e:
                logger.warning(f"Error capturando frame {i+1}: {e}")
                continue

        logger.info("Capturando nube de puntos e imagen infrarroja...")

        # Obtener frame de profundidad
        inDepth = qDepth.get()
        depthFrame = inDepth.getFrame().astype(np.float32)

        # Calibración de cámara (valores aproximados, deberías leerlos de device.readCalibration())
        calib = device.readCalibration()
        intrinsics = calib.getCameraIntrinsics(dai.CameraBoardSocket.LEFT, depthFrame.shape[1], depthFrame.shape[0])
        fx, fy = intrinsics[0][0], intrinsics[1][1]
        cx, cy = intrinsics[0][2], intrinsics[1][2]

        # Convertir depth → nube de puntos
        points = []
        for y in range(depthFrame.shape[0]):
            for x in range(depthFrame.shape[1]):
                Z = depthFrame[y, x]
                if Z > 0:
                    X = (x - cx) * Z / fx
                    Y = (y - cy) * Z / fy
                    points.append([X, Y, Z])
        points = np.array(points)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        filename = f"{node}_{timestamp}"

        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            ply_filename = f"{filename}.ply"
            if o3d.io.write_point_cloud(f"{data_directory}/{ply_filename}", pcd, write_ascii=False):
                logger.info(f"[PLY] Guardado como: {ply_filename}")
            else:
                logger.error(f"No se pudo guardar el archivo PLY: {ply_filename}")
        except Exception:
            logger.exception("Fallo al capturar o guardar la nube de puntos")

        # Guardar imagen IR
        try:
            monoFrame = qMonoRight.get().getCvFrame()
            ir_filename = f"{filename}_ir.jpg"
            if cv2.imwrite(f"{data_directory}/{ir_filename}", monoFrame):
                logger.info(f"[Imagen IR] Guardada como: {ir_filename}")
            else:
                logger.error(f"No se pudo guardar la imagen IR: {ir_filename}")
        except Exception:
            logger.exception("Fallo al capturar o guardar la imagen infrarroja")

except Exception:
    logger.critical("No se pudo iniciar el dispositivo o el pipeline", exc_info=True)

logger.info("Ejecucción finalizada.")
