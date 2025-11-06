import depthai as dai
import numpy as np
import cv2
from datetime import datetime
import os
import logging
from logging.handlers import RotatingFileHandler
from collections import defaultdict

DOT_PROJECTOR_INTENSITY = 1
FLOOD_LIGHT_INTENSITY = 0
FPS = 5
FRAME_TARGET = 50
RESOLUTION = dai.MonoCameraProperties.SensorResolution.THE_400_P

# Número de nubes consecutivas que se combinarán antes de guardar el archivo.
FRAMES_TO_ACCUMULATE = int(os.getenv("FRAMES_TO_ACCUMULATE", 5))

# Límites del volumen a conservar (en metros)
CROP_X_MIN = float(os.getenv("CROP_X_MIN", "-2.6"))
CROP_X_MAX = float(os.getenv("CROP_X_MAX", "2.6"))
CROP_Y_MIN = float(os.getenv("CROP_Y_MIN", "-2.6"))
CROP_Y_MAX = float(os.getenv("CROP_Y_MAX", "2.6"))
CROP_Z_MIN = float(os.getenv("CROP_Z_MIN", "-5.2"))
CROP_Z_MAX = float(os.getenv("CROP_Z_MAX", "0.5"))

# Parámetros del filtrado estadístico
NB_NEIGHBORS = int(os.getenv("NB_NEIGHBORS", "60"))
STD_RATIO = float(os.getenv("STD_RATIO", "1.0"))
VOXEL_SIZE = float(os.getenv("VOXEL_SIZE", "0.05"))

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
data_directory = os.getenv('DATA_DIRECTORY_EXPORT', '/home/pi/Documentos')
image_infra = os.getenv('IMAGE_IR', 'False').lower() == 'true'

logger.info("Inicializando pipeline y nodos de cámara...")

# Crear pipeline
pipeline = dai.Pipeline()

# Crear nodos - SIN el nodo PointCloud para evitar segfault en RPi3
mono_left = pipeline.create(dai.node.MonoCamera)
mono_right = pipeline.create(dai.node.MonoCamera)
depth = pipeline.create(dai.node.StereoDepth)

# XLink outputs
xout_depth = pipeline.create(dai.node.XLinkOut)
xout_depth.setStreamName("depth")

xout_mono_right = pipeline.create(dai.node.XLinkOut)
xout_mono_right.setStreamName("mono_right")

logger.info("Configurando cámaras mono...")

# Configuración de cámaras
mono_resolution = RESOLUTION
mono_left.setResolution(mono_resolution)
mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
mono_left.setFps(FPS)

mono_right.setResolution(mono_resolution)
mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
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
config.postProcessing.speckleFilter.speckleRange = 25
config.postProcessing.spatialFilter.enable = True
config.postProcessing.spatialFilter.holeFillingRadius = 40
config.postProcessing.spatialFilter.numIterations = 5
depth.initialConfig.set(config)

depth.setDepthAlign(dai.CameraBoardSocket.CAM_B)

logger.info("Enlazando nodos del pipeline...")

mono_left.out.link(depth.left)
mono_right.out.link(depth.right)
depth.depth.link(xout_depth.input)
mono_right.out.link(xout_mono_right.input)

logger.info("Pipeline configurado correctamente.")


def depth_to_pointcloud_cropped(depth_map, calibration_data):
    """
    Convierte depth map a nube de puntos aplicando el crop directamente
    para ahorrar memoria en RPi3.
    """
    h, w = depth_map.shape
    
    intrinsics = calibration_data.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, 
                                                       dai.Size2f(w, h))
    
    fx = intrinsics[0][0]
    fy = intrinsics[1][1]
    cx = intrinsics[0][2]
    cy = intrinsics[1][2]
    
    # Crear mallas de coordenadas
    u = np.arange(w)
    v = np.arange(h)
    u, v = np.meshgrid(u, v)
    
    u = u.flatten()
    v = v.flatten()
    z = depth_map.flatten().astype(np.float32)
    
    # Filtrar puntos con profundidad inválida
    valid = (z > 0) & (z < 5000)
    u = u[valid]
    v = v[valid]
    z = z[valid]
    
    # Convertir a coordenadas 3D (en mm primero)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # Convertir a metros
    x = x / 1000.0
    y = y / 1000.0
    z = z / 1000.0
    
    # Invertir Z
    z = -z
    
    # Aplicar crop INMEDIATAMENTE para reducir memoria
    crop_mask = (
        (x >= CROP_X_MIN) & (x <= CROP_X_MAX) &
        (y >= CROP_Y_MIN) & (y <= CROP_Y_MAX) &
        (z >= CROP_Z_MIN) & (z <= CROP_Z_MAX)
    )
    
    points = np.stack([x[crop_mask], y[crop_mask], z[crop_mask]], axis=1).astype(np.float32)
    return points


def voxel_downsample(points, voxel_size):
    """
    Downsampling por voxel sin usar Open3D.
    Agrupa puntos en celdas 3D y toma el promedio de cada celda.
    """
    if len(points) == 0:
        return points
    
    # Convertir coordenadas a índices de voxel
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)
    
    # Crear diccionario para agrupar puntos por voxel
    voxel_dict = defaultdict(list)
    for i, idx in enumerate(voxel_indices):
        key = (idx[0], idx[1], idx[2])
        voxel_dict[key].append(points[i])
    
    # Calcular centroide de cada voxel
    downsampled = []
    for voxel_points in voxel_dict.values():
        centroid = np.mean(voxel_points, axis=0)
        downsampled.append(centroid)
    
    return np.array(downsampled, dtype=np.float32)


def statistical_outlier_removal(points, nb_neighbors, std_ratio):
    """
    Filtrado estadístico de outliers sin usar Open3D.
    Elimina puntos cuya distancia promedio a sus vecinos es anómala.
    """
    if len(points) < nb_neighbors:
        return points
    
    from scipy.spatial import cKDTree
    
    # Construir árbol KD para búsqueda de vecinos
    tree = cKDTree(points)
    
    # Para cada punto, encontrar sus k vecinos más cercanos
    distances, _ = tree.query(points, k=nb_neighbors + 1)
    
    # Ignorar la distancia a sí mismo (primera columna)
    distances = distances[:, 1:]
    
    # Calcular distancia promedio de cada punto a sus vecinos
    mean_distances = np.mean(distances, axis=1)
    
    # Calcular umbral estadístico
    global_mean = np.mean(mean_distances)
    global_std = np.std(mean_distances)
    threshold = global_mean + std_ratio * global_std
    
    # Filtrar puntos que están dentro del umbral
    mask = mean_distances < threshold
    return points[mask]


def write_ply(filename, points):
    """
    Escribe un archivo PLY manualmente sin usar Open3D.
    """
    with open(filename, 'wb') as f:
        # Header
        header = f"""ply
format binary_little_endian 1.0
element vertex {len(points)}
property float x
property float y
property float z
end_header
"""
        f.write(header.encode('ascii'))
        
        # Datos binarios
        points.astype(np.float32).tofile(f)
    
    return True


try:
    with dai.Device(pipeline) as device:
        logger.info("Dispositivo conectado y pipeline cargado.")
        
        calibration_data = device.readCalibration()
        
        q_depth = device.getOutputQueue(name="depth", maxSize=1, blocking=True)
        q_mono_right = device.getOutputQueue(name="mono_right", maxSize=1, blocking=True)
        
        logger.info("Configurando proyector IR...")
        device.setIrLaserDotProjectorIntensity(DOT_PROJECTOR_INTENSITY)
        device.setIrFloodLightIntensity(FLOOD_LIGHT_INTENSITY)
        
        logger.info("Esperando %s frames para estabilizar...", FRAME_TARGET)
        for i in range(FRAME_TARGET):
            try:
                q_depth.get()
                q_mono_right.get()
            except Exception as exc:
                logger.warning("Error capturando frame %s: %s", i + 1, exc)
                continue
        
        logger.info("Capturando nube de puntos e imagen infrarroja...")
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        filename = f'{node}_{timestamp}'
        
        try:
            # Capturar múltiples depth maps
            accumulated_points = []
            for frame_idx in range(max(1, FRAMES_TO_ACCUMULATE)):
                try:
                    in_depth = q_depth.get()
                    depth_frame = in_depth.getFrame()
                    
                    # Convertir y cropear en un solo paso para ahorrar memoria
                    points = depth_to_pointcloud_cropped(depth_frame, calibration_data)
                    accumulated_points.append(points)
                    
                    logger.debug("Frame %s/%s: %s puntos (ya cropeados)", 
                               frame_idx + 1, FRAMES_TO_ACCUMULATE, len(points))
                except Exception as exc:
                    logger.warning("Error obteniendo frame %s/%s: %s",
                                 frame_idx + 1, FRAMES_TO_ACCUMULATE, exc)
                    continue
            
            if not accumulated_points:
                raise RuntimeError("No se pudieron obtener frames de nube de puntos")
            
            # Fusionar todas las nubes
            merged_points = np.concatenate(accumulated_points, axis=0)
            logger.info("Total de puntos fusionados (cropeados): %s", len(merged_points))
            
            if len(merged_points) == 0:
                raise RuntimeError("No quedan puntos tras aplicar el recorte")
            
            # Downsampling manual
            logger.info("Aplicando downsampling con voxel_size=%s...", VOXEL_SIZE)
            down_points = voxel_downsample(merged_points, VOXEL_SIZE)
            logger.info("Puntos después del downsampling: %s", len(down_points))
            
            if len(down_points) == 0:
                raise RuntimeError("No quedan puntos después del downsampling")
            
            # Filtrado estadístico manual
            logger.info("Aplicando filtrado estadístico...")
            try:
                filtered_points = statistical_outlier_removal(
                    down_points,
                    nb_neighbors=max(1, NB_NEIGHBORS),
                    std_ratio=max(0.1, STD_RATIO)
                )
                logger.info("Puntos después del filtrado: %s", len(filtered_points))
            except ImportError:
                logger.warning("scipy no disponible, omitiendo filtrado estadístico")
                filtered_points = down_points
            
            if len(filtered_points) == 0:
                raise RuntimeError("No quedan puntos tras el filtrado estadístico")
            
            # Guardar archivo PLY manualmente
            ply_filename = f"{filename}.ply"
            ply_path = os.path.join(data_directory, ply_filename)
            
            if write_ply(ply_path, filtered_points):
                logger.info("[PLY] Guardado como: %s (%s puntos)", ply_filename, len(filtered_points))
            else:
                logger.error("No se pudo guardar el archivo PLY: %s", ply_filename)
                
        except Exception:
            logger.exception("Fallo al capturar o guardar la nube de puntos")
        
        # Capturar imagen infrarroja
        try:
            mono_frame = q_mono_right.get().getCvFrame()
            ir_filename = f"{filename}_ir.jpg"
            ir_path = os.path.join(data_directory, ir_filename)
            
            if cv2.imwrite(ir_path, mono_frame):
                logger.info("[Imagen IR] Guardada como: %s", ir_filename)
            else:
                logger.error("No se pudo guardar la imagen IR: %s", ir_filename)
        except Exception:
            logger.exception("Fallo al capturar o guardar la imagen infrarroja")

except Exception:
    logger.critical("No se pudo iniciar el dispositivo o el pipeline", exc_info=True)

logger.info("Ejecución finalizada.")
logger.info("")