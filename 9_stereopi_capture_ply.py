"""Captura nubes de puntos con StereoPi v2 y las guarda como archivos PLY binarios."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from picamera import PiCamera


# Parámetros configurables --------------------------------------------------
OUTPUT_DIR = Path(os.getenv("STEREOPI_PLY_DIR", "ply_stereopi"))
SETTINGS_FILE = Path(os.getenv("STEREOPI_SETTINGS", "3dmap_set.txt"))
CALIBRATION_TEMPLATE = Path(os.getenv("STEREOPI_CALIB_TEMPLATE", "./calibration_data/{}p/stereo_camera_calibration.npz"))
LOG_DIR = Path(os.getenv("STEREOPI_LOG_DIR", "logs"))

CAM_WIDTH = int(os.getenv("STEREOPI_CAM_WIDTH", 1280))
CAM_HEIGHT = int(os.getenv("STEREOPI_CAM_HEIGHT", 480))
FRAME_RATE = int(os.getenv("STEREOPI_FRAME_RATE", 20))
SCALE_RATIO = float(os.getenv("STEREOPI_SCALE_RATIO", 0.5))

PLY_PATTERN = "captura_{:03d}.ply"


# Utilidades ----------------------------------------------------------------
def setup_logger() -> logging.Logger:
    LOG_DIR.mkdir(exist_ok=True)
    log_filename = LOG_DIR / f"silo_register_{datetime.now().strftime('%Y%m%d')}.log"

    logger = logging.getLogger("silo_register")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    file_handler = RotatingFileHandler(log_filename, maxBytes=5 * 1024 * 1024, backupCount=10, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)


def next_ply_index() -> int:
    ensure_output_dir()
    counter = 0
    for file in sorted(OUTPUT_DIR.glob("*.ply")):
        stem = file.stem
        if stem.startswith("captura_"):
            try:
                idx = int(stem.split("_")[1])
            except (IndexError, ValueError):
                continue
            counter = max(counter, idx + 1)
    return counter


def load_map_settings(filename: Path) -> dict:
    if not filename.is_file():
        raise FileNotFoundError(
            f"Stereo BM preset '{filename}' not found. Ejecuta 5_dm_tune.py para generarlo."
        )
    with filename.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {
        "SADWindowSize": data.get("SADWindowSize", 5),
        "preFilterSize": data.get("preFilterSize", 5),
        "preFilterCap": data.get("preFilterCap", 29),
        "minDisparity": data.get("minDisparity", 0),
        "numberOfDisparities": data.get("numberOfDisparities", 64),
        "textureThreshold": data.get("textureThreshold", 10),
        "uniquenessRatio": data.get("uniquenessRatio", 15),
        "speckleRange": data.get("speckleRange", 0),
        "speckleWindowSize": data.get("speckleWindowSize", 0),
    }


def configure_matcher(settings: dict) -> cv2.StereoBM:
    block_size = settings["SADWindowSize"]
    if block_size % 2 == 0:
        block_size += 1
    block_size = max(5, block_size)

    matcher = cv2.StereoBM_create(
        numDisparities=settings["numberOfDisparities"],
        blockSize=block_size,
    )
    matcher.setPreFilterType(1)
    matcher.setPreFilterSize(settings["preFilterSize"])
    matcher.setPreFilterCap(settings["preFilterCap"])
    matcher.setMinDisparity(settings["minDisparity"])
    matcher.setTextureThreshold(settings["textureThreshold"])
    matcher.setUniquenessRatio(settings["uniquenessRatio"])
    matcher.setSpeckleRange(settings["speckleRange"])
    matcher.setSpeckleWindowSize(settings["speckleWindowSize"])
    return matcher


def load_calibration(height: int) -> Tuple[np.ndarray, ...]:
    scaled_height = int(height * SCALE_RATIO)
    calib_path = Path(str(CALIBRATION_TEMPLATE).format(scaled_height))
    if not calib_path.is_file():
        raise FileNotFoundError(
            f"Archivo de calibración '{calib_path}' no encontrado. Ejecuta 4_calibration_fisheye.py primero."
        )
    data = np.load(str(calib_path))
    return (
        data["leftMapX"],
        data["leftMapY"],
        data["rightMapX"],
        data["rightMapY"],
        data["dispartityToDepthMap"],
    )


def split_and_rectify(frame: np.ndarray, left_map_x, left_map_y, right_map_x, right_map_y):
    bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, width = gray.shape
    mid = width // 2

    left_raw_gray = gray[:, :mid]
    right_raw_gray = gray[:, mid:]
    left_raw_color = bgr[:, :mid]
    right_raw_color = bgr[:, mid:]

    left_rectified = cv2.remap(left_raw_gray, left_map_x, left_map_y, interpolation=cv2.INTER_LINEAR)
    right_rectified = cv2.remap(right_raw_gray, right_map_x, right_map_y, interpolation=cv2.INTER_LINEAR)

    left_rectified_color = cv2.remap(left_raw_color, left_map_x, left_map_y, interpolation=cv2.INTER_LINEAR)
    right_rectified_color = cv2.remap(right_raw_color, right_map_x, right_map_y, interpolation=cv2.INTER_LINEAR)

    return (
        left_rectified,
        right_rectified,
        left_rectified_color,
        right_rectified_color,
    )


def compute_point_cloud(matcher: cv2.StereoBM, left: np.ndarray, right: np.ndarray, qq: np.ndarray):
    disparity_raw = matcher.compute(left, right)
    disparity = disparity_raw.astype(np.float32) / 16.0
    points = cv2.reprojectImageTo3D(disparity, qq)
    mask = (disparity > disparity.min()) & np.isfinite(points[:, :, 2])
    return disparity, points, mask


def save_point_cloud(path: Path, points: np.ndarray, mask: np.ndarray) -> bool:
    masked_points = points[mask]
    if masked_points.size == 0:
        return False

    vertices = np.ascontiguousarray(masked_points.astype(np.float32))
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )

    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        vertices.tofile(handle)

    return True


def start_camera() -> Tuple[PiCamera, np.ndarray]:
    cam_width = int((CAM_WIDTH + 31) / 32) * 32
    cam_height = int((CAM_HEIGHT + 15) / 16) * 16
    img_width = int(cam_width * SCALE_RATIO)
    img_height = int(cam_height * SCALE_RATIO)

    camera = PiCamera(stereo_mode="side-by-side", stereo_decimate=False)
    camera.resolution = (cam_width, cam_height)
    camera.framerate = FRAME_RATE
    time.sleep(2)
    buffer = np.empty((img_height, img_width, 4), dtype=np.uint8)
    return camera, buffer


def main() -> None:
    logger = setup_logger()
    ensure_output_dir()

    logger.info("Cargando parámetros del stereo matcher...")
    settings = load_map_settings(SETTINGS_FILE)
    matcher = configure_matcher(settings)

    logger.info("Cargando datos de calibración...")
    left_map_x, left_map_y, right_map_x, right_map_y, qq = load_calibration(CAM_HEIGHT)

    logger.info("Inicializando cámara StereoPi...")
    camera, buffer = start_camera()

    ply_index = next_ply_index()
    logger.info("Captura iniciada. Presiona 's' para guardar PLY, 'q' para salir.")

    try:
        while True:
            camera.capture(buffer, format="bgra", resize=(buffer.shape[1], buffer.shape[0]))
            frame = buffer.copy()

            (
                left_rectified,
                right_rectified,
                left_rectified_color,
                right_rectified_color,
            ) = split_and_rectify(frame, left_map_x, left_map_y, right_map_x, right_map_y)

            disparity, points, mask = compute_point_cloud(matcher, left_rectified, right_rectified, qq)
            disparity_norm = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            disparity_color = cv2.applyColorMap(disparity_norm.astype(np.uint8), cv2.COLORMAP_JET)

            cv2.imshow("StereoPi Left", left_rectified_color)
            cv2.imshow("StereoPi Right", right_rectified_color)
            cv2.imshow("StereoPi Disparity", disparity_color)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                ply_path = OUTPUT_DIR / PLY_PATTERN.format(ply_index)
                if save_point_cloud(ply_path, points, mask):
                    logger.info("PLY guardado: %s", ply_path)
                    print(f"PLY guardado: {ply_path}")
                    ply_index += 1
                else:
                    logger.warning("No se pudo guardar la nube de puntos: nube vacía")

            elif key == ord("q"):
                logger.info("Salida solicitada por el usuario.")
                break

    except KeyboardInterrupt:
        logger.info("Captura interrumpida por el usuario.")
    finally:
        camera.close()
        cv2.destroyAllWindows()
        logger.info("Cámara cerrada. Programa finalizado.")


if __name__ == "__main__":
    main()