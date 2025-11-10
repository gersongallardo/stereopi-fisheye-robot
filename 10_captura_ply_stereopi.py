"""Enhanced PLY capture pipeline for StereoPi v2."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from picamera import PiCamera

OUTPUT_DIR = Path("pointclouds")
CALIBRATION_TEMPLATE = "./calibration_data/{}p/stereo_camera_calibration.npz"

CAM_WIDTH = 1280
CAM_HEIGHT = 480
FRAME_RATE = 20
SCALE_RATIO = 0.5

ACCUMULATED_FRAMES = 5
GAUSSIAN_KERNEL = (5, 5)
CONTRAST_CLIP_LIMIT = 2.0
CONTRAST_GRID_SIZE = (8, 8)

NUM_DISPARITIES = 160
BLOCK_SIZE = 5
WLS_LAMBDA = 8000.0
WLS_SIGMA = 1.2


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / timestamp
    run_dir.mkdir()
    return run_dir


def load_calibration(height: int) -> Tuple[np.ndarray, ...]:
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


def capture_stereo_frames(count: int) -> np.ndarray:
    cam_width = int((CAM_WIDTH + 31) / 32) * 32
    cam_height = int((CAM_HEIGHT + 15) / 16) * 16
    img_width = int(cam_width * SCALE_RATIO)
    img_height = int(cam_height * SCALE_RATIO)

    camera = PiCamera(stereo_mode="side-by-side", stereo_decimate=False)
    camera.resolution = (cam_width, cam_height)
    camera.framerate = FRAME_RATE

    buffer = np.zeros((img_height, img_width, 4), dtype=np.uint8)
    accumulator = np.zeros_like(buffer, dtype=np.float32)

    try:
        for _ in range(count):
            camera.capture(buffer, format="bgra", resize=(img_width, img_height))
            accumulator += buffer.astype(np.float32)
    finally:
        camera.close()

    averaged = (accumulator / float(count)).astype(np.uint8)
    return averaged


def split_and_rectify(
    frame: np.ndarray,
    left_map_x: np.ndarray,
    left_map_y: np.ndarray,
    right_map_x: np.ndarray,
    right_map_y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    left_color, right_color = np.split(bgr, 2, axis=1)

    clahe = cv2.createCLAHE(clipLimit=CONTRAST_CLIP_LIMIT, tileGridSize=CONTRAST_GRID_SIZE)
    left_gray = clahe.apply(cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY))
    right_gray = clahe.apply(cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY))

    left_gray = cv2.GaussianBlur(left_gray, GAUSSIAN_KERNEL, 0)
    right_gray = cv2.GaussianBlur(right_gray, GAUSSIAN_KERNEL, 0)

    left_rectified = cv2.remap(left_gray, left_map_x, left_map_y, interpolation=cv2.INTER_LINEAR)
    right_rectified = cv2.remap(right_gray, right_map_x, right_map_y, interpolation=cv2.INTER_LINEAR)

    left_rectified_color = cv2.remap(
        left_color, left_map_x, left_map_y, interpolation=cv2.INTER_LINEAR
    )
    right_rectified_color = cv2.remap(
        right_color, right_map_x, right_map_y, interpolation=cv2.INTER_LINEAR
    )

    return (
        left_rectified,
        right_rectified,
        left_rectified_color,
        right_rectified_color,
        left_color,
        right_color,
    )


def configure_matchers() -> Tuple[cv2.StereoMatcher, Optional[cv2.StereoMatcher], Optional[object]]:
    block_size = max(3, BLOCK_SIZE)
    block_size += 1 - block_size % 2

    matcher_left = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=NUM_DISPARITIES,
        blockSize=block_size,
        P1=8 * 3 * block_size**2,
        P2=32 * 3 * block_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=5,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    try:
        ximgproc = cv2.ximgproc  # type: ignore[attr-defined]
    except AttributeError:
        return matcher_left, None, None

    matcher_right = ximgproc.createRightMatcher(matcher_left)
    wls_filter = ximgproc.createDisparityWLSFilter(matcher_left)
    wls_filter.setLambda(WLS_LAMBDA)
    wls_filter.setSigmaColor(WLS_SIGMA)

    return matcher_left, matcher_right, wls_filter


def compute_disparity(
    left_matcher: cv2.StereoMatcher,
    right_matcher: Optional[cv2.StereoMatcher],
    wls_filter: Optional[object],
    left_rectified: np.ndarray,
    right_rectified: np.ndarray,
    guidance: np.ndarray,
) -> np.ndarray:
    left_disp = left_matcher.compute(left_rectified, right_rectified).astype(np.float32) / 16.0

    if right_matcher is None or wls_filter is None:
        return left_disp

    right_disp = right_matcher.compute(right_rectified, left_rectified).astype(np.float32) / 16.0
    filtered = wls_filter.filter(left_disp, guidance, disparity_map_right=right_disp)
    return filtered


def compute_point_cloud(disparity: np.ndarray, qq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    disparity = cv2.medianBlur(disparity, 5)
    points = cv2.reprojectImageTo3D(disparity, qq)
    mask = np.isfinite(points[:, :, 2]) & (disparity > disparity.min())
    return points, mask


def save_point_cloud(path: Path, points: np.ndarray, colors: np.ndarray, mask: np.ndarray) -> None:
    masked_points = points[mask]
    masked_colors = colors[mask]

    ply_data = np.hstack([masked_points, masked_colors])
    header = "ply\nformat ascii 1.0\n"
    header += f"element vertex {len(ply_data)}\n"
    header += "property float x\nproperty float y\nproperty float z\n"
    header += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
    header += "end_header\n"

    with path.open("w", encoding="utf-8") as handle:
        handle.write(header)
        np.savetxt(handle, ply_data, fmt="%.5f %.5f %.5f %d %d %d")


def main() -> None:
    run_dir = ensure_output_dir()

    (
        left_map_x,
        left_map_y,
        right_map_x,
        right_map_y,
        qq,
        _,
    ) = load_calibration(CAM_HEIGHT)

    frame = capture_stereo_frames(ACCUMULATED_FRAMES)

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

    left_matcher, right_matcher, wls_filter = configure_matchers()
    disparity = compute_disparity(
        left_matcher,
        right_matcher,
        wls_filter,
        left_rectified,
        right_rectified,
        left_rectified_color,
    )

    disp_norm = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disp_color = cv2.applyColorMap(disp_norm.astype(np.uint8), cv2.COLORMAP_TURBO)
    cv2.imwrite(str(run_dir / "disparity.png"), disp_color)

    points, mask = compute_point_cloud(disparity, qq)
    save_point_cloud(run_dir / "densified_point_cloud.ply", points, left_rectified_color, mask)

    print(f"Saved enhanced capture assets to {run_dir}")


if __name__ == "__main__":
    main()
