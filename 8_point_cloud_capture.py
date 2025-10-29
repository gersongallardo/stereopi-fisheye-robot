"""Capture a single StereoPi v2 frame and export a colored point cloud."""

from datetime import datetime
import json
from pathlib import Path

import cv2
import numpy as np
from picamera import PiCamera


OUTPUT_DIR = Path("pointclouds")
SETTINGS_FILE = Path("3dmap_set.txt")
CALIBRATION_TEMPLATE = "./calibration_data/{}p/stereo_camera_calibration.npz"

CAM_WIDTH = 1280
CAM_HEIGHT = 480
FRAME_RATE = 20
SCALE_RATIO = 0.5


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / timestamp
    run_dir.mkdir()
    return run_dir


def load_map_settings(filename: Path) -> dict:
    if not filename.is_file():
        raise FileNotFoundError(
            f"Stereo BM preset '{filename}' not found. Run 5_dm_tune.py to create it."
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


def load_calibration(height: int):
    scaled_height = int(height * SCALE_RATIO)
    calib_path = Path(CALIBRATION_TEMPLATE.format(scaled_height))
    if not calib_path.is_file():
        raise FileNotFoundError(
            f"Calibration file '{calib_path}' not found. Run 4_calibration_fisheye.py first."
        )
    data = np.load(str(calib_path))
    return (
        data["leftMapX"],
        data["leftMapY"],
        data["rightMapX"],
        data["rightMapY"],
        data["dispartityToDepthMap"],
        tuple(data["imageSize"]),
    )


def capture_stereo_frame() -> np.ndarray:
    cam_width = int((CAM_WIDTH + 31) / 32) * 32
    cam_height = int((CAM_HEIGHT + 15) / 16) * 16
    img_width = int(cam_width * SCALE_RATIO)
    img_height = int(cam_height * SCALE_RATIO)

    camera = PiCamera(stereo_mode="side-by-side", stereo_decimate=False)
    camera.resolution = (cam_width, cam_height)
    camera.framerate = FRAME_RATE

    buffer = np.zeros((img_height, img_width, 4), dtype=np.uint8)
    try:
        camera.capture(buffer, format="bgra", resize=(img_width, img_height))
    finally:
        camera.close()
    return buffer


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
        left_raw_color,
        right_raw_color,
    )


def compute_point_cloud(matcher: cv2.StereoBM, left: np.ndarray, right: np.ndarray, qq: np.ndarray):
    disparity_raw = matcher.compute(left, right)
    disparity = disparity_raw.astype(np.float32) / 16.0
    points = cv2.reprojectImageTo3D(disparity, qq)
    mask = disparity > disparity.min()
    mask &= np.isfinite(points[:, :, 2])
    return disparity, points, mask


def save_point_cloud(path: Path, points: np.ndarray, colors: np.ndarray, mask: np.ndarray) -> None:
    masked_points = points[mask]
    masked_colors = colors[mask]
    ply_data = np.hstack([masked_points, masked_colors])
    header = """ply\nformat ascii 1.0\n"""
    header += f"element vertex {len(ply_data)}\n"
    header += "property float x\nproperty float y\nproperty float z\n"
    header += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
    header += "end_header\n"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(header)
        np.savetxt(handle, ply_data, fmt="%.4f %.4f %.4f %d %d %d")


def main():
    run_dir = ensure_output_dir()
    settings = load_map_settings(SETTINGS_FILE)
    matcher = configure_matcher(settings)

    left_map_x, left_map_y, right_map_x, right_map_y, qq, _ = load_calibration(CAM_HEIGHT)

    frame = capture_stereo_frame()
    (
        left_rectified,
        right_rectified,
        left_rectified_color,
        right_rectified_color,
        left_raw_color,
        right_raw_color,
    ) = split_and_rectify(frame, left_map_x, left_map_y, right_map_x, right_map_y)

    cv2.imwrite(str(run_dir / "raw_left.jpg"), left_raw_color)
    cv2.imwrite(str(run_dir / "raw_right.jpg"), right_raw_color)
    cv2.imwrite(str(run_dir / "rectified_left.jpg"), left_rectified)
    cv2.imwrite(str(run_dir / "rectified_right.jpg"), right_rectified)
    cv2.imwrite(str(run_dir / "rectified_left_color.jpg"), left_rectified_color)
    cv2.imwrite(str(run_dir / "rectified_right_color.jpg"), right_rectified_color)

    disparity, points, mask = compute_point_cloud(matcher, left_rectified, right_rectified, qq)
    disparity_norm = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disparity_color = cv2.applyColorMap(disparity_norm.astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(str(run_dir / "disparity.png"), disparity_color)

    save_point_cloud(run_dir / "point_cloud.ply", points, left_rectified_color, mask)
    print(f"Saved capture assets to {run_dir}")


if __name__ == "__main__":
    main()
