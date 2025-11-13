"""Optimized PLY capture for StereoPi v2 - Sin dependencia de Open3D."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from picamera import PiCamera

# ==================== CONFIGURACIÓN ====================
OUTPUT_DIR = Path("/home/pi/pi1")

# Ruta base del proyecto StereoPi
STEREOPI_BASE = Path("/home/pi/stereopi-fisheye-robot")
CALIBRATION_DIR = STEREOPI_BASE / "calibration_data"
CALIBRATION_HEIGHT = int(os.getenv("CALIBRATION_HEIGHT", 240))

# Configuración de cámara - IMPORTANTE: debe coincidir con la calibración
CAM_WIDTH = 1280
CAM_HEIGHT = 480
FRAME_RATE = 15

# Resolución objetivo para los resultados y la nube de puntos
DEFAULT_TARGET_WIDTH = 640
DEFAULT_TARGET_HEIGHT = 400

# ==================== PARÁMETROS MEJORADOS ====================

# Captura de frames (más frames = menos ruido)
ACCUMULATED_FRAMES = 50  # Aumentado de 30 a 50

# Preprocesamiento mejorado
GAUSSIAN_KERNEL = (5, 5)  # Kernel más grande para mejor suavizado
CONTRAST_CLIP_LIMIT = 2.0  # Mejor contraste
CONTRAST_GRID_SIZE = (8, 8)

# Parámetros de disparidad - OPTIMIZADOS PARA MÁS DENSIDAD
NUM_DISPARITIES = 128  # Reducido para mejor velocidad/calidad
BLOCK_SIZE = 5  # Aumentado para mejor matching
WLS_LAMBDA = 8000.0  # Aumentado para mejor filtrado
WLS_SIGMA = 1.5  # Aumentado para suavizado

# SGBM mejorado
UNIQUENESS_RATIO = 10  # Más selectivo
SPECKLE_WINDOW_SIZE = 100  # Ventana más grande
SPECKLE_RANGE = 32  # Rango más amplio
PRE_FILTER_CAP = 63
DISP12_MAX_DIFF = 1

# ==================== FILTRADO MANUAL (sin Open3D) ====================
# Filtrado de profundidad - MÁS PERMISIVO
MIN_DEPTH_MM = 100  # 10 cm mínimo (era 300)
MAX_DEPTH_MM = 20000  # 20 metros máximo (era 8000)

# Filtrado de outliers basado en vecindad
ENABLE_NEIGHBOR_FILTER = True
NEIGHBOR_RADIUS = 5  # Radio para buscar vecinos (píxeles)
MIN_NEIGHBORS = 3  # Mínimo de vecinos válidos requeridos

# Downsampling manual
DOWNSAMPLE_FACTOR = 2  # Factor de reducción (2 = mitad de puntos)

# Filtrado por rango de disparidad
MIN_DISPARITY = 0.5  # Disparidad mínima válida
MAX_DISPARITY = 127.0  # Disparidad máxima válida

# Configuración del nodo
NODE = int(os.getenv('NODE', 0))

_XIMGPROC_SPEC = importlib.util.find_spec("cv2.ximgproc")
XIMGPROC_MODULE = importlib.import_module("cv2.ximgproc") if _XIMGPROC_SPEC else None

# ==================== CONFIGURACIÓN DE LOGGING ====================
log_directory = OUTPUT_DIR / 'logs'
log_directory.mkdir(parents=True, exist_ok=True)

log_filename = log_directory / f'stereopi_capture_{datetime.now().strftime("%Y%m%d")}.log'

logger = logging.getLogger('stereopi_capture')
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    file_handler = RotatingFileHandler(
        str(log_filename),
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

    logger.info("Archivo de log diario: %s", log_filename)


def _parse_resolution(env_name: str, default_value: int) -> int:
    value = os.getenv(env_name)
    if value is None:
        return default_value

    try:
        parsed = int(value)
        if parsed > 0:
            return parsed
    except ValueError:
        logger.warning("Valor inválido para %s=%s, usando %d", env_name, value, default_value)

    return default_value


def _parse_int(env_name: str, default_value: int) -> int:
    value = os.getenv(env_name)
    if value is None:
        return default_value

    try:
        return int(value)
    except ValueError:
        logger.warning(
            "Valor inválido para %s=%s, usando %d", env_name, value, default_value
        )
        return default_value


TARGET_WIDTH = _parse_resolution("TARGET_WIDTH", DEFAULT_TARGET_WIDTH)
TARGET_HEIGHT = _parse_resolution("TARGET_HEIGHT", DEFAULT_TARGET_HEIGHT)
CAMERA_ROTATION = _parse_int("CAMERA_ROTATION", 0)


def resolve_output_size(calibration_size: Tuple[int, int]) -> Tuple[int, int]:
    """Determina el tamaño final de salida respetando la relación de aspecto."""
    calib_width, calib_height = calibration_size
    width = TARGET_WIDTH
    height = TARGET_HEIGHT

    if width <= 0 and height <= 0:
        return calib_width, calib_height

    if width <= 0:
        width = int(round(height * calib_width / calib_height))
    elif height <= 0:
        height = int(round(width * calib_height / calib_width))

    return width, height


def ensure_output_dir() -> None:
    """Crea el directorio de salida si no existe."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Directorio de salida: %s", OUTPUT_DIR)


def load_calibration() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
    """Carga los datos de calibración de la cámara."""
    calib_subdir = CALIBRATION_DIR / f"{CALIBRATION_HEIGHT}p"
    calib_path = calib_subdir / "stereo_camera_calibration.npz"
    
    if not calib_path.is_file():
        logger.error("Archivo de calibración no encontrado: %s", calib_path)
        raise FileNotFoundError(f"Calibration file '{calib_path}' not found.")
    
    logger.info("Cargando calibración desde: %s", calib_path)
    data = np.load(str(calib_path))
    
    required_keys = ['leftMapX', 'leftMapY', 'rightMapX', 'rightMapY', 'dispartityToDepthMap', 'imageSize']
    missing_keys = [key for key in required_keys if key not in data.files]
    if missing_keys:
        logger.error("Faltan claves en el archivo de calibración: %s", missing_keys)
        raise ValueError(f"Calibration file is missing keys: {missing_keys}")
    
    image_size = tuple(int(v) for v in data["imageSize"])
    logger.info("Calibración cargada correctamente para tamaño: %s", image_size)
    
    return (
        data["leftMapX"],
        data["leftMapY"],
        data["rightMapX"],
        data["rightMapY"],
        data["dispartityToDepthMap"],
        image_size,
    )


def capture_stereo_frames(count: int) -> np.ndarray:
    """Captura y promedia múltiples frames estéreo."""
    logger.info("Inicializando cámara StereoPi...")

    cam_width = int((CAM_WIDTH + 31) / 32) * 32
    cam_height = int((CAM_HEIGHT + 15) / 16) * 16

    camera = PiCamera(stereo_mode="side-by-side", stereo_decimate=False)
    camera.resolution = (cam_width, cam_height)
    camera.framerate = FRAME_RATE
    rotation = CAMERA_ROTATION % 360
    camera.rotation = rotation
    
    # Ajustes de exposición para mejor calidad
    camera.exposure_mode = 'auto'
    camera.awb_mode = 'auto'
    
    if rotation:
        logger.info("Rotación de cámara aplicada: %d°", rotation)

    logger.info("Resolución de cámara: %dx%d", cam_width, cam_height)
    logger.info("Capturando %d frames para promediar...", count)
    
    buffer = np.zeros((cam_height, cam_width, 4), dtype=np.uint8)
    accumulator = np.zeros_like(buffer, dtype=np.float32)
    
    try:
        # Dar tiempo a la cámara para estabilizarse
        import time
        time.sleep(2)
        
        for i in range(count):
            camera.capture(buffer, format="bgra")
            accumulator += buffer.astype(np.float32)
            logger.debug("Frame %d/%d capturado", i + 1, count)
    finally:
        camera.close()
    
    averaged = (accumulator / float(count)).astype(np.uint8)
    logger.info("Frames promediados correctamente")
    return averaged


def split_and_rectify(
    frame: np.ndarray,
    left_map_x: np.ndarray,
    left_map_y: np.ndarray,
    right_map_x: np.ndarray,
    right_map_y: np.ndarray,
    calibration_size: Tuple[int, int],
    output_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[float, float]]:
    """Divide el frame estéreo y rectifica las imágenes."""
    logger.info("Procesando y rectificando imágenes estéreo...")
    
    calib_width, calib_height = calibration_size
    out_width, out_height = output_size

    bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    left_color, right_color = np.split(bgr, 2, axis=1)
    
    left_color_calib = cv2.resize(left_color, (calib_width, calib_height), interpolation=cv2.INTER_AREA)
    right_color_calib = cv2.resize(right_color, (calib_width, calib_height), interpolation=cv2.INTER_AREA)
    
    left_gray = cv2.cvtColor(left_color_calib, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_color_calib, cv2.COLOR_BGR2GRAY)
    
    # CLAHE mejorado
    clahe = cv2.createCLAHE(clipLimit=CONTRAST_CLIP_LIMIT, tileGridSize=CONTRAST_GRID_SIZE)
    left_gray = clahe.apply(left_gray)
    right_gray = clahe.apply(right_gray)
    
    # Suavizado
    left_gray = cv2.GaussianBlur(left_gray, GAUSSIAN_KERNEL, 0)
    right_gray = cv2.GaussianBlur(right_gray, GAUSSIAN_KERNEL, 0)

    # Rectificar
    left_rectified = cv2.remap(left_gray, left_map_x, left_map_y, interpolation=cv2.INTER_LINEAR)
    right_rectified = cv2.remap(right_gray, right_map_x, right_map_y, interpolation=cv2.INTER_LINEAR)
    left_rectified_color = cv2.remap(
        left_color_calib, left_map_x, left_map_y, interpolation=cv2.INTER_LINEAR
    )

    scale_x = out_width / float(calib_width)
    scale_y = out_height / float(calib_height)

    if (out_width, out_height) != (calib_width, calib_height):
        logger.info(
            "Escalando imágenes de %dx%d a %dx%d",
            calib_width, calib_height, out_width, out_height
        )
        left_rectified = cv2.resize(left_rectified, (out_width, out_height), interpolation=cv2.INTER_CUBIC)
        right_rectified = cv2.resize(right_rectified, (out_width, out_height), interpolation=cv2.INTER_CUBIC)
        left_rectified_color = cv2.resize(
            left_rectified_color, (out_width, out_height), interpolation=cv2.INTER_LINEAR
        )

    logger.info("Rectificación completada")
    return left_rectified, right_rectified, left_rectified_color, (scale_x, scale_y)


def configure_matchers() -> Tuple[cv2.StereoMatcher, cv2.StereoMatcher, object]:
    """Configura los matchers SGBM con filtro WLS mejorado."""
    logger.info("Configurando matchers estéreo...")

    block_size = max(3, BLOCK_SIZE)
    block_size += 1 - block_size % 2
    
    # Matcher izquierdo con parámetros MEJORADOS
    matcher_left = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=NUM_DISPARITIES,
        blockSize=block_size,
        P1=8 * 3 * block_size**2,
        P2=32 * 3 * block_size**2,
        disp12MaxDiff=DISP12_MAX_DIFF,
        uniquenessRatio=UNIQUENESS_RATIO,
        speckleWindowSize=SPECKLE_WINDOW_SIZE,
        speckleRange=SPECKLE_RANGE,
        preFilterCap=PRE_FILTER_CAP,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    
    if XIMGPROC_MODULE is not None:
        matcher_right = XIMGPROC_MODULE.createRightMatcher(matcher_left)
        wls_filter = XIMGPROC_MODULE.createDisparityWLSFilter(matcher_left)
        wls_filter.setLambda(WLS_LAMBDA)
        wls_filter.setSigmaColor(WLS_SIGMA)
        logger.info("Filtro WLS activado (lambda=%.1f, sigma=%.1f)", WLS_LAMBDA, WLS_SIGMA)
    else:
        logger.warning("Módulo cv2.ximgproc no disponible")
        matcher_right = None
        wls_filter = None
    
    logger.info("Matchers configurados: numDisparities=%d, blockSize=%d", NUM_DISPARITIES, block_size)
    return matcher_left, matcher_right, wls_filter


def enhance_disparity_without_wls(disparity: np.ndarray) -> np.ndarray:
    """Aplica un filtrado alternativo cuando ximgproc no está disponible."""
    refined = disparity.copy()

    disparity_16s = np.round(refined * 16.0).astype(np.int16)
    cv2.filterSpeckles(disparity_16s, 0, SPECKLE_WINDOW_SIZE, SPECKLE_RANGE)
    refined = disparity_16s.astype(np.float32) / 16.0

    refined = cv2.medianBlur(refined, 5)
    refined = cv2.bilateralFilter(refined, 9, 75, 75)

    return refined


def compute_disparity(
    left_matcher: cv2.StereoMatcher,
    right_matcher: cv2.StereoMatcher,
    wls_filter: object,
    left_rectified: np.ndarray,
    right_rectified: np.ndarray,
    guidance: np.ndarray,
) -> np.ndarray:
    """Calcula el mapa de disparidad con filtrado WLS."""
    logger.info("Calculando mapa de disparidad...")
    
    left_disp = left_matcher.compute(left_rectified, right_rectified).astype(np.float32) / 16.0
    
    if right_matcher is not None and wls_filter is not None:
        right_disp = right_matcher.compute(right_rectified, left_rectified).astype(np.float32) / 16.0
        filtered = wls_filter.filter(left_disp, guidance, disparity_map_right=right_disp)
    else:
        filtered = enhance_disparity_without_wls(left_disp)
        logger.info("Filtrado alternativo aplicado")
    
    valid_disparities = filtered[filtered > 0]
    if len(valid_disparities) > 0:
        logger.info("Disparidad - Min: %.2f, Max: %.2f, Media: %.2f", 
                    np.min(valid_disparities), np.max(valid_disparities), np.mean(valid_disparities))
    else:
        logger.warning("No se encontraron disparidades válidas")
    
    return filtered


def scale_q_matrix(qq: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    """Escala la matriz Q de reproyección."""
    scaled = qq.astype(np.float64).copy()
    scaled[0, 3] *= scale_x
    scaled[1, 3] *= scale_y
    scaled[2, 3] *= scale_x
    scaled[3, 3] *= scale_x
    return scaled


def compute_point_cloud(disparity: np.ndarray, qq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Genera la nube de puntos 3D desde el mapa de disparidad."""
    logger.info("Generando nube de puntos 3D...")
    
    # Suavizado adicional
    disparity = cv2.medianBlur(disparity, 5)
    
    # Reprojectar a 3D
    points = cv2.reprojectImageTo3D(disparity, qq)
    
    # Máscara de puntos válidos - MÁS PERMISIVA
    mask = (
        np.isfinite(points[:, :, 0]) & 
        np.isfinite(points[:, :, 1]) & 
        np.isfinite(points[:, :, 2]) & 
        (disparity >= MIN_DISPARITY) &
        (disparity <= MAX_DISPARITY) &
        (np.abs(points[:, :, 2]) > MIN_DEPTH_MM) &  # Profundidad mínima
        (np.abs(points[:, :, 2]) < MAX_DEPTH_MM)   # Profundidad máxima
    )
    
    valid_points = np.count_nonzero(mask)
    logger.info("Puntos válidos en la nube INICIAL: %d", valid_points)
    
    if valid_points == 0:
        logger.warning("  NO HAY PUNTOS VÁLIDOS. Estadísticas de profundidad:")
        z_values = points[:, :, 2][np.isfinite(points[:, :, 2])]
        if len(z_values) > 0:
            logger.warning("    Z min: %.2f mm, max: %.2f mm", z_values.min(), z_values.max())
        disp_valid = disparity[disparity > 0]
        if len(disp_valid) > 0:
            logger.warning("    Disparidad min: %.2f, max: %.2f", disp_valid.min(), disp_valid.max())
    
    return points, mask


def apply_neighbor_filter(mask: np.ndarray, radius: int = NEIGHBOR_RADIUS, min_neighbors: int = MIN_NEIGHBORS) -> np.ndarray:
    """Filtra puntos que no tienen suficientes vecinos válidos."""
    if not ENABLE_NEIGHBOR_FILTER:
        return mask
    
    logger.info("Aplicando filtro de vecindad (radio=%d, min=%d)...", radius, min_neighbors)
    
    # Convertir máscara a uint8
    mask_uint8 = mask.astype(np.uint8)
    
    # Contar vecinos válidos usando convolución
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2+1, radius*2+1))
    neighbor_count = cv2.filter2D(mask_uint8, -1, kernel)
    
    # Mantener solo puntos con suficientes vecinos
    filtered_mask = (neighbor_count >= min_neighbors) & mask
    
    before = np.count_nonzero(mask)
    after = np.count_nonzero(filtered_mask)
    logger.info("Filtro vecindad: %d -> %d puntos (eliminados: %d)", before, after, before - after)
    
    return filtered_mask


def downsample_points(points: np.ndarray, colors: np.ndarray, mask: np.ndarray, factor: int = DOWNSAMPLE_FACTOR) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce la densidad de puntos mediante submuestreo."""
    if factor <= 1:
        return points[mask], colors[mask]
    
    logger.info("Aplicando downsampling (factor=%d)...", factor)
    
    # Crear máscara de submuestreo
    h, w = mask.shape
    downsample_mask = np.zeros_like(mask)
    downsample_mask[::factor, ::factor] = True
    
    # Combinar con máscara de puntos válidos
    final_mask = mask & downsample_mask
    
    before = np.count_nonzero(mask)
    after = np.count_nonzero(final_mask)
    logger.info("Downsampling: %d -> %d puntos", before, after)
    
    return points[final_mask], colors[final_mask]


def filter_by_density_clustering(points: np.ndarray, colors: np.ndarray, distance_threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Clustering manual simple basado en densidad espacial (sin sklearn/open3d)."""
    if len(points) == 0:
        return points, colors
    
    logger.info("Aplicando clustering manual de densidad...")
    
    # Convertir a metros
    points_m = points / 1000.0
    
    # Encontrar el centroide de todos los puntos
    centroid = np.mean(points_m, axis=0)
    
    # Calcular distancias al centroide
    distances = np.linalg.norm(points_m - centroid, axis=1)
    
    # Calcular percentil 90 como umbral adaptativo
    threshold = np.percentile(distances, 90)
    threshold = max(threshold, distance_threshold)
    
    # Mantener solo puntos cercanos al centroide
    mask = distances < threshold * 2  # Factor 2 para ser más permisivo
    
    before = len(points)
    after = np.count_nonzero(mask)
    logger.info("Clustering: %d -> %d puntos (umbral=%.2fm)", before, after, threshold)
    
    return points_m[mask], colors[mask]


def save_ply_binary(
    path: Path, 
    points: np.ndarray, 
    colors: np.ndarray
) -> None:
    """Guarda la nube de puntos en formato PLY binario."""
    logger.info("Guardando nube de puntos en formato PLY binario...")
    
    num_points = len(points)
    
    if num_points == 0:
        logger.error("No hay puntos válidos para guardar")
        return
    
    header = f"""ply
format binary_little_endian 1.0
element vertex {num_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    
    with path.open("wb") as f:
        f.write(header.encode('ascii'))
        
        for i in range(num_points):
            # XYZ como float32
            f.write(points[i, 0].astype(np.float32).tobytes())
            f.write(points[i, 1].astype(np.float32).tobytes())
            f.write(points[i, 2].astype(np.float32).tobytes())
            # RGB (BGR a RGB)
            f.write(colors[i, 2].astype(np.uint8).tobytes())  # R
            f.write(colors[i, 1].astype(np.uint8).tobytes())  # G
            f.write(colors[i, 0].astype(np.uint8).tobytes())  # B
    
    logger.info("[PLY] Guardado: %s (%d puntos)", path.name, num_points)


def main() -> None:
    """Función principal de captura."""
    logger.info("=" * 60)
    logger.info("StereoPi v2 - Versión MEJORADA (sin Open3D)")
    logger.info("=" * 60)
    
    try:
        ensure_output_dir()
        
        # Cargar calibración
        (
            left_map_x,
            left_map_y,
            right_map_x,
            right_map_y,
            qq,
            calibration_size,
        ) = load_calibration()

        output_size = resolve_output_size(calibration_size)
        logger.info(
            "Resolución final: %dx%d (calibración %dx%d)",
            output_size[0], output_size[1],
            calibration_size[0], calibration_size[1]
        )

        # Capturar frames
        frame = capture_stereo_frames(ACCUMULATED_FRAMES)

        # Procesar imágenes
        (
            left_rectified,
            right_rectified,
            left_rectified_color,
            (scale_x, scale_y),
        ) = split_and_rectify(
            frame,
            left_map_x,
            left_map_y,
            right_map_x,
            right_map_y,
            calibration_size,
            output_size,
        )
        
        # Configurar matchers
        left_matcher, right_matcher, wls_filter = configure_matchers()
        
        # Calcular disparidad
        disparity = compute_disparity(
            left_matcher,
            right_matcher,
            wls_filter,
            left_rectified,
            right_rectified,
            left_rectified_color,
        )
        
        # Generar nube de puntos
        scaled_q = scale_q_matrix(qq, scale_x, scale_y)
        points, mask = compute_point_cloud(disparity, scaled_q)
        
        if np.count_nonzero(mask) == 0:
            logger.error(" No se generaron puntos válidos. Abortando guardado de PLY.")
            logger.error("   Revisa la calibración y las condiciones de iluminación.")
        else:
            # Aplicar filtros manuales
            mask = apply_neighbor_filter(mask)
            
            # Downsample y clustering
            points_filtered, colors_filtered = downsample_points(points, left_rectified_color, mask)
            
            # Convertir a metros e invertir Z
            points_filtered = points_filtered / 1000.0
            points_filtered[:, 2] *= -1
            
            # Clustering simple
            points_filtered, colors_filtered = filter_by_density_clustering(points_filtered, colors_filtered)
            
            # Generar nombres de archivo
            timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            filename_base = f'{NODE}_{timestamp}'
            
            # Guardar PLY
            ply_path = OUTPUT_DIR / f"{filename_base}.ply"
            save_ply_binary(ply_path, points_filtered, colors_filtered)
        
        # Guardar imagen IR (siempre)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        filename_base = f'{NODE}_{timestamp}'
        ir_path = OUTPUT_DIR / f"{filename_base}_ir.jpg"
        if cv2.imwrite(str(ir_path), left_rectified):
            logger.info("[Imagen IR] Guardada: %s", ir_path.name)
        else:
            logger.error("No se pudo guardar la imagen IR")
        
        logger.info("=" * 60)
        logger.info("✓ Captura completada")
        logger.info("✓ Archivos guardados en: %s", OUTPUT_DIR)
        logger.info("=" * 60)
        
    except FileNotFoundError as e:
        logger.error("Error de archivo: %s", e)
    except Exception:
        logger.exception("Error crítico durante la captura")
    
    logger.info("Ejecución finalizada.")


if __name__ == "__main__":
    main()