"""Optimized PLY capture for StereoPi v2 - Output similar to OAK camera."""

from __future__ import annotations

import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from picamera import PiCamera

# ==================== CONFIGURACIÓN ====================
OUTPUT_DIR = Path("/home/pi1")

# Ruta base del proyecto StereoPi
STEREOPI_BASE = Path("/home/pi/stereopi-fisheye-robot")
CALIBRATION_DIR = STEREOPI_BASE / "calibration_data"
CALIBRATION_HEIGHT = int(os.getenv("CALIBRATION_HEIGHT", 240))

# Configuración de cámara - IMPORTANTE: debe coincidir con la calibración
CAM_WIDTH = 1280
CAM_HEIGHT = 480
FRAME_RATE = 15

# Resolución objetivo para los resultados y la nube de puntos.
# Se puede sobreescribir con variables de entorno TARGET_WIDTH y TARGET_HEIGHT
DEFAULT_TARGET_WIDTH = 640  # Coincide con OAK-D 400p (ancho)
DEFAULT_TARGET_HEIGHT = 400

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



# Captura de frames
ACCUMULATED_FRAMES = 3

# Preprocesamiento
GAUSSIAN_KERNEL = (3, 3)
CONTRAST_CLIP_LIMIT = 1.5
CONTRAST_GRID_SIZE = (8, 8)

# Parámetros de disparidad - Optimizados para más densidad
NUM_DISPARITIES = 256  # Debe ser múltiplo de 16
BLOCK_SIZE = 3
WLS_LAMBDA = 4000.0
WLS_SIGMA = 0.8

# Configuración del nodo
NODE = int(os.getenv('NODE', 0))

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


TARGET_WIDTH = _parse_resolution("TARGET_WIDTH", DEFAULT_TARGET_WIDTH)
TARGET_HEIGHT = _parse_resolution("TARGET_HEIGHT", DEFAULT_TARGET_HEIGHT)


# ==================== FUNCIONES ====================

def ensure_output_dir() -> None:
    """Crea el directorio de salida si no existe."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Directorio de salida: %s", OUTPUT_DIR)


def load_calibration() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
    """Carga los datos de calibración de la cámara."""
    # El archivo de calibración está organizado por altura (ej. 240p)
    calib_subdir = CALIBRATION_DIR / f"{CALIBRATION_HEIGHT}p"
    calib_path = calib_subdir / "stereo_camera_calibration.npz"
    
    if not calib_path.is_file():
        logger.error("Archivo de calibración no encontrado: %s", calib_path)
        logger.error("Por favor ejecuta primero: python3 4_calibration_fisheye.py")
        logger.error("Asegúrate que img_height=%d en el script de calibración", CALIBRATION_HEIGHT)
        raise FileNotFoundError(f"Calibration file '{calib_path}' not found.")
    
    logger.info("Cargando calibración desde: %s", calib_path)
    data = np.load(str(calib_path))
    
    # Verificar que el archivo contiene los datos necesarios
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
    
    # Alinear a múltiplos requeridos por la cámara
    cam_width = int((CAM_WIDTH + 31) / 32) * 32
    cam_height = int((CAM_HEIGHT + 15) / 16) * 16
    
    camera = PiCamera(stereo_mode="side-by-side", stereo_decimate=False)
    camera.resolution = (cam_width, cam_height)
    camera.framerate = FRAME_RATE
    
    # Rotar 180 grados para corregir orientación
    camera.rotation = 180
    
    logger.info("Resolución de cámara: %dx%d", cam_width, cam_height)
    logger.info("Capturando %d frames para promediar...", count)
    
    # Capturar en resolución completa primero
    buffer = np.zeros((cam_height, cam_width, 4), dtype=np.uint8)
    accumulator = np.zeros_like(buffer, dtype=np.float32)
    
    try:
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

    # Convertir a BGR y dividir en izquierda y derecha
    bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    left_color, right_color = np.split(bgr, 2, axis=1)
    
    # Ajustar a la resolución de calibración
    left_color_calib = cv2.resize(left_color, (calib_width, calib_height), interpolation=cv2.INTER_AREA)
    right_color_calib = cv2.resize(right_color, (calib_width, calib_height), interpolation=cv2.INTER_AREA)
    
    # Convertir a escala de grises
    left_gray = cv2.cvtColor(left_color_calib, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_color_calib, cv2.COLOR_BGR2GRAY)
    
    # Aplicar CLAHE para mejorar contraste
    clahe = cv2.createCLAHE(clipLimit=CONTRAST_CLIP_LIMIT, tileGridSize=CONTRAST_GRID_SIZE)
    left_gray = clahe.apply(left_gray)
    right_gray = clahe.apply(right_gray)
    
    # Suavizado ligero
    left_gray = cv2.GaussianBlur(left_gray, GAUSSIAN_KERNEL, 0)
    right_gray = cv2.GaussianBlur(right_gray, GAUSSIAN_KERNEL, 0)

    # Rectificar imágenes usando los mapas de calibración
    left_rectified = cv2.remap(left_gray, left_map_x, left_map_y, interpolation=cv2.INTER_LINEAR)
    right_rectified = cv2.remap(right_gray, right_map_x, right_map_y, interpolation=cv2.INTER_LINEAR)

    left_rectified_color = cv2.remap(
        left_color_calib, left_map_x, left_map_y, interpolation=cv2.INTER_LINEAR
    )

    scale_x = out_width / float(calib_width)
    scale_y = out_height / float(calib_height)

    if (out_width, out_height) != (calib_width, calib_height):
        logger.info(
            "Escalando imágenes rectificadas de %dx%d a %dx%d",
            calib_width,
            calib_height,
            out_width,
            out_height,
        )
        left_rectified = cv2.resize(left_rectified, (out_width, out_height), interpolation=cv2.INTER_CUBIC)
        right_rectified = cv2.resize(right_rectified, (out_width, out_height), interpolation=cv2.INTER_CUBIC)
        left_rectified_color = cv2.resize(
            left_rectified_color, (out_width, out_height), interpolation=cv2.INTER_LINEAR
        )
    logger.info("Factor de escala aplicado: %.2fx horizontal, %.2fx vertical", scale_x, scale_y)

    logger.info("Rectificación completada")
    return left_rectified, right_rectified, left_rectified_color, (scale_x, scale_y)


def configure_matchers() -> Tuple[cv2.StereoMatcher, cv2.StereoMatcher, object]:
    """Configura los matchers SGBM con filtro WLS."""
    logger.info("Configurando matchers estéreo...")
    
    block_size = max(3, BLOCK_SIZE)
    block_size += 1 - block_size % 2
    
    # Matcher izquierdo con parámetros optimizados
    matcher_left = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=NUM_DISPARITIES,
        blockSize=block_size,
        P1=8 * 3 * block_size**2,
        P2=32 * 3 * block_size**2,
        disp12MaxDiff=2,
        uniquenessRatio=5,
        speckleWindowSize=50,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    
    # Matcher derecho y filtro WLS
    try:
        ximgproc = cv2.ximgproc
        matcher_right = ximgproc.createRightMatcher(matcher_left)
        wls_filter = ximgproc.createDisparityWLSFilter(matcher_left)
        wls_filter.setLambda(WLS_LAMBDA)
        wls_filter.setSigmaColor(WLS_SIGMA)
        logger.info("Filtro WLS activado")
    except AttributeError:
        logger.warning("cv2.ximgproc no disponible, usando solo matcher izquierdo")
        matcher_right = None
        wls_filter = None
    
    logger.info("Matchers configurados: numDisparities=%d, blockSize=%d", NUM_DISPARITIES, block_size)
    return matcher_left, matcher_right, wls_filter


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
        filtered = left_disp
    
    valid_disparities = filtered[filtered > 0]
    if len(valid_disparities) > 0:
        logger.info("Disparidad calculada - Min: %.2f, Max: %.2f, Media: %.2f", 
                    np.min(valid_disparities), np.max(valid_disparities), np.mean(valid_disparities))
    else:
        logger.warning("No se encontraron disparidades válidas")
    
    return filtered


def scale_q_matrix(qq: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    """Escala la matriz Q de reproyección cuando cambian ancho y alto."""

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
    
    # Crear máscara de puntos válidos
    # Filtrar puntos con coordenadas válidas y dentro de un rango razonable
    mask = (
        np.isfinite(points[:, :, 0]) & 
        np.isfinite(points[:, :, 1]) & 
        np.isfinite(points[:, :, 2]) & 
        (disparity > 0) &
        (np.abs(points[:, :, 2]) < 10000)  # Filtrar puntos muy lejanos (10 metros)
    )
    
    valid_points = np.count_nonzero(mask)
    logger.info("Puntos válidos en la nube: %d", valid_points)
    
    return points, mask


def save_ply_binary(
    path: Path, 
    points: np.ndarray, 
    colors: np.ndarray, 
    mask: np.ndarray
) -> None:
    """Guarda la nube de puntos en formato PLY binario (similar a OAK)."""
    logger.info("Guardando nube de puntos en formato PLY binario...")
    
    # Extraer puntos y colores válidos
    masked_points = points[mask]
    masked_colors = colors[mask]
    
    # Convertir a metros (los puntos están en milímetros)
    masked_points = masked_points / 1000.0
    
    # Invertir eje Z como en OAK
    masked_points[:, 2] *= -1
    
    num_points = len(masked_points)
    
    if num_points == 0:
        logger.error("No hay puntos válidos para guardar")
        return
    
    # Formato binario PLY
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
        
        # Escribir datos en formato binario
        for i in range(num_points):
            # XYZ como float32
            f.write(masked_points[i, 0].astype(np.float32).tobytes())
            f.write(masked_points[i, 1].astype(np.float32).tobytes())
            f.write(masked_points[i, 2].astype(np.float32).tobytes())
            # RGB como uint8 (BGR a RGB)
            f.write(masked_colors[i, 2].astype(np.uint8).tobytes())  # R
            f.write(masked_colors[i, 1].astype(np.uint8).tobytes())  # G
            f.write(masked_colors[i, 0].astype(np.uint8).tobytes())  # B
    
    logger.info("[PLY] Guardado como: %s (%d puntos)", path.name, num_points)


def main() -> None:
    """Función principal de captura."""
    logger.info("=" * 60)
    logger.info("Iniciando captura StereoPi v2")
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
            "Resolución final deseada: %dx%d (calibración %dx%d)",
            output_size[0],
            output_size[1],
            calibration_size[0],
            calibration_size[1],
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
        
        # Generar nombres de archivo con timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        filename_base = f'{NODE}_{timestamp}'
        
        # Guardar PLY binario
        ply_path = OUTPUT_DIR / f"{filename_base}.ply"
        save_ply_binary(ply_path, points, left_rectified_color, mask)
        
        # Guardar imagen IR (rectificada izquierda en escala de grises)
        ir_path = OUTPUT_DIR / f"{filename_base}_ir.jpg"
        if cv2.imwrite(str(ir_path), left_rectified):
            logger.info("[Imagen IR] Guardada como: %s", ir_path.name)
        else:
            logger.error("No se pudo guardar la imagen IR")
        
        logger.info("=" * 60)
        logger.info("Captura completada exitosamente")
        logger.info("Archivos guardados en: %s", OUTPUT_DIR)
        logger.info("=" * 60)
        
    except FileNotFoundError as e:
        logger.error("Error de archivo: %s", e)
    except Exception:
        logger.exception("Error crítico durante la captura")
    
    logger.info("Ejecución finalizada.")
    logger.info("")


if __name__ == "__main__":
    main()
