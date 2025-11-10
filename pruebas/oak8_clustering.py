import depthai as dai
import numpy as np
import cv2
from datetime import datetime
import os
import logging
from logging.handlers import RotatingFileHandler
import open3d as o3d

DOT_PROJECTOR_INTENSITY = 1
FLOOD_LIGHT_INTENSITY = 0.3
FPS = 5
FRAME_TARGET = 50
RESOLUTION = dai.MonoCameraProperties.SensorResolution.THE_400_P

# Parámetros de filtrado (clustering)
ENABLE_CLUSTERING = True  # Filtrado por clustering (recomendado para silos)
CLUSTER_EPS = 0.18  # Distancia máxima entre puntos del mismo cluster (metros)
CLUSTER_MIN_POINTS = 80  # Mínimo de puntos para formar un cluster

#para el tamaño del archivo
DOWNSAMPLE_VOXEL_SIZE = 0.04  # Tamaño del voxel para reducir la densidad de puntos

#outliers
ENABLE_STATISTICAL_FILTER = True  # Filtro estadístico adicional
STAT_NB_NEIGHBORS = 20  # Número de vecinos a considerar
STAT_STD_RATIO = 2.0  # Ratio de desviación estándar

# Crear el directorio de logs si no existe
log_directory = os.path.join(os.getcwd(), 'logs')
os.makedirs(log_directory, exist_ok=True)

# Nombre del archivo de log con fecha
log_filename = os.path.join(log_directory, f'silo_register_{datetime.now().strftime("%Y%m%d")}.log')

# Configurar el logger
logger = logging.getLogger('silo_register')
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler = RotatingFileHandler(
        log_filename,
        maxBytes=5 * 1024 * 1024,
        backupCount=10,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# Configuración
logger.info('Importando variables...')
node = int(os.getenv('NODE', 0))
data_directory = os.getenv('DATA_DIRECTORY_EXPORT', '/home/pi/oak5')
image_infra = os.getenv('IMAGE_IR', 'False').lower() == 'true'

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
mono_resolution = RESOLUTION
mono_left.setResolution(mono_resolution)
mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
#mono_left.setCamera("left")
mono_left.setFps(FPS)

mono_right.setResolution(mono_resolution)
mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
#mono_right.setCamera("right")
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
config.postProcessing.thresholdFilter.minRange = 200
config.postProcessing.thresholdFilter.maxRange = 5000
config.postProcessing.speckleFilter.enable = True
config.postProcessing.speckleFilter.speckleRange = 28
config.postProcessing.spatialFilter.enable = True
config.postProcessing.spatialFilter.holeFillingRadius = 2
config.postProcessing.spatialFilter.numIterations = 2
depth.initialConfig.set(config)

#depth.setDepthAlign(dai.CameraBoardSocket.LEFT)
depth.setDepthAlign(dai.CameraBoardSocket.CAM_B)

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
            logger.info("Nube de puntos capturada con %s puntos", len(pcd.points))
            
            # ========== FILTRADO DE RUIDO ==========
            
            # 1. Downsampling para reducir puntos (PRIMERO)
            logger.info("Aplicando downsampling...")
            pcd = pcd.voxel_down_sample(voxel_size=DOWNSAMPLE_VOXEL_SIZE)
            logger.info(f"Puntos después de downsampling: {len(pcd.points)}")
            
            # 2. Filtro estadístico de outliers (SEGUNDO - rápido)
            if ENABLE_STATISTICAL_FILTER and len(pcd.points) > 0:
                logger.info("Aplicando filtro estadístico de outliers...")
                pcd, ind = pcd.remove_statistical_outlier(
                    nb_neighbors=STAT_NB_NEIGHBORS,
                    std_ratio=STAT_STD_RATIO
                )
                logger.info(f"Puntos después de filtro estadístico: {len(pcd.points)}")
            
            # 3. Filtrado por clustering (ÚLTIMO - ahora sobre menos puntos)
            if ENABLE_CLUSTERING and len(pcd.points) > 0:
                logger.info("Aplicando filtrado por clustering...")
                labels = np.array(pcd.cluster_dbscan(
                    eps=CLUSTER_EPS, 
                    min_points=CLUSTER_MIN_POINTS, 
                    print_progress=False
                ))
                
                # Contar puntos por cluster
                unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
                
                if len(unique_labels) > 0:
                    logger.info(f"Se encontraron {len(unique_labels)} clusters válidos")
                    
                    # Mantener solo el cluster más grande (asumiendo que es el silo)
                    largest_cluster_idx = unique_labels[np.argmax(counts)]
                    largest_cluster_size = counts.max()
                    
                    logger.info(f"Cluster más grande: {largest_cluster_idx} con {largest_cluster_size} puntos")
                    
                    # Filtrar puntos
                    mask = labels == largest_cluster_idx
                    pcd = pcd.select_by_index(np.where(mask)[0])
                    logger.info(f"Puntos después de clustering: {len(pcd.points)}")
                else:
                    logger.warning("No se encontraron clusters válidos")
            
            # ========================================
            
            # Guardar nube de puntos filtrada
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

logger.info("Ejecución finalizada.")
logger.info("")