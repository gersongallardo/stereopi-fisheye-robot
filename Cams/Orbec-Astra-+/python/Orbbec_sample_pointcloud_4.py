#!/usr/bin/env python3
"""Point-cloud capture sample for the Orbbec Astra+ camera using ctypes bindings.

This script reproduces the functionality of the C Sample-PointCloud example
bundled with the Orbbec SDK. It opens the depth and (optionally) color streams,
configures depth-to-color alignment when available, and generates PLY files for
depth-only and RGBD point clouds.

Usage examples:
    python sample_pointcloud.py --mode both
    python sample_pointcloud.py --mode rgbd --output-dir ./captures
"""
from __future__ import annotations

import argparse
import ctypes
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Constants mirrored from the Orbbec SDK headers.
# ---------------------------------------------------------------------------
OB_SENSOR_COLOR = 2
OB_SENSOR_DEPTH = 3

OB_FORMAT_POINT = 19
OB_FORMAT_RGB_POINT = 20

OB_PROFILE_DEFAULT = 0

ALIGN_DISABLE = 0
ALIGN_D2C_HW_MODE = 1
ALIGN_D2C_SW_MODE = 2

OB_LOG_SEVERITY_ERROR = 3

DEFAULT_TIMEOUT_MS = 100

DEFAULT_OUTPUT_DIR = Path("/home/gerson/Documentos/Camara orbbec")
DEFAULT_LIBRARY_PATH = Path("/home/gerson/OrbbecSDK/lib/linux_x64/libOrbbecSDK.so")
DEFAULT_SILO_ID = 1
DEFAULT_VOXEL_SIZE = 0.05

# ---------------------------------------------------------------------------
# ctypes representations of the structures required by the sample.
# ---------------------------------------------------------------------------
class ErrorHandle(ctypes.c_void_p):
    """Wrapper type for ob_error*."""


class OBCameraIntrinsic(ctypes.Structure):
    _fields_ = [
        ("fx", ctypes.c_float),
        ("fy", ctypes.c_float),
        ("cx", ctypes.c_float),
        ("cy", ctypes.c_float),
        ("width", ctypes.c_int16),
        ("height", ctypes.c_int16),
    ]


class OBCameraDistortion(ctypes.Structure):
    _fields_ = [
        ("k1", ctypes.c_float),
        ("k2", ctypes.c_float),
        ("k3", ctypes.c_float),
        ("k4", ctypes.c_float),
        ("k5", ctypes.c_float),
        ("k6", ctypes.c_float),
        ("p1", ctypes.c_float),
        ("p2", ctypes.c_float),
    ]


class OBD2CTransform(ctypes.Structure):
    _fields_ = [
        ("rot", ctypes.c_float * 9),
        ("trans", ctypes.c_float * 3),
    ]


class OBCameraParam(ctypes.Structure):
    _fields_ = [
        ("depthIntrinsic", OBCameraIntrinsic),
        ("rgbIntrinsic", OBCameraIntrinsic),
        ("depthDistortion", OBCameraDistortion),
        ("rgbDistortion", OBCameraDistortion),
        ("transform", OBD2CTransform),
        ("isMirrored", ctypes.c_bool),
    ]


class OBPoint(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float), ("z", ctypes.c_float)]


class OBColorPoint(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("z", ctypes.c_float),
        ("r", ctypes.c_float),
        ("g", ctypes.c_float),
        ("b", ctypes.c_float),
    ]


# ---------------------------------------------------------------------------
# Exceptions and helpers.
# ---------------------------------------------------------------------------
class OrbbecError(RuntimeError):
    """Raised when the Orbbec SDK reports an error."""


def _platform_library_name() -> List[str]:
    if sys.platform.startswith("linux"):
        return ["libOrbbecSDK.so"]
    if sys.platform == "darwin":
        return ["libOrbbecSDK.dylib"]
    if os.name == "nt":
        return ["OrbbecSDK.dll"]
    return ["libOrbbecSDK.so"]


class OrbbecSDK:
    """Minimal ctypes wrapper around the C API used by the sample."""

    def __init__(self, library_path: Optional[Path] = None) -> None:
        self._lib = self._load_library(library_path)
        self._configure_signatures()

    # ------------------------------------------------------------------
    # Library loading / setup helpers.
    # ------------------------------------------------------------------
    def _load_library(self, user_path: Optional[Path]) -> ctypes.CDLL:
        search_paths: List[str] = []

        if user_path:
            search_paths.append(str(user_path))

        env_path = os.getenv("ORBBEC_LIB_PATH")
        if env_path:
            search_paths.append(env_path)

        script_root = Path(__file__).resolve().parents[1]
        packaged = script_root / "lib"
        if packaged.exists():
            if sys.platform.startswith("linux"):
                search_paths.append(str(packaged / "linux_x64" / "libOrbbecSDK.so"))
            elif os.name == "nt":
                search_paths.append(str(packaged / "win_x64" / "OrbbecSDK.dll"))

        search_paths.extend(_platform_library_name())

        load_errors: List[str] = []
        for candidate in search_paths:
            try:
                return ctypes.CDLL(candidate)
            except OSError as exc:  # pragma: no cover - best-effort diagnostics
                load_errors.append(f"{candidate}: {exc}")
        msg = "\n".join(load_errors) or "No candidates were tried"
        raise OrbbecError(
            "Unable to load libOrbbecSDK. Provide --library-path or set ORBBEC_LIB_PATH.\n"
            f"Tried:\n{msg}"
        )

    def _configure_signatures(self) -> None:
        lib = self._lib
        err_ptr = ctypes.POINTER(ErrorHandle)

        # Error helpers
        lib.ob_error_message.argtypes = [ErrorHandle]
        lib.ob_error_message.restype = ctypes.c_char_p
        lib.ob_error_function.argtypes = [ErrorHandle]
        lib.ob_error_function.restype = ctypes.c_char_p
        lib.ob_error_args.argtypes = [ErrorHandle]
        lib.ob_error_args.restype = ctypes.c_char_p
        lib.ob_error_exception_type.argtypes = [ErrorHandle]
        lib.ob_error_exception_type.restype = ctypes.c_int
        lib.ob_delete_error.argtypes = [ErrorHandle]
        lib.ob_delete_error.restype = None

        lib.ob_set_logger_severity.argtypes = [ctypes.c_int, err_ptr]
        lib.ob_set_logger_severity.restype = None

        lib.ob_create_pipeline.argtypes = [err_ptr]
        lib.ob_create_pipeline.restype = ctypes.c_void_p
        lib.ob_delete_pipeline.argtypes = [ctypes.c_void_p, err_ptr]
        lib.ob_delete_pipeline.restype = None

        lib.ob_create_config.argtypes = [err_ptr]
        lib.ob_create_config.restype = ctypes.c_void_p
        lib.ob_delete_config.argtypes = [ctypes.c_void_p, err_ptr]
        lib.ob_delete_config.restype = None

        lib.ob_pipeline_get_stream_profile_list.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            err_ptr,
        ]
        lib.ob_pipeline_get_stream_profile_list.restype = ctypes.c_void_p

        lib.ob_stream_profile_list_get_profile.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            err_ptr,
        ]
        lib.ob_stream_profile_list_get_profile.restype = ctypes.c_void_p

        lib.ob_video_stream_profile_fps.argtypes = [ctypes.c_void_p, err_ptr]
        lib.ob_video_stream_profile_fps.restype = ctypes.c_uint32

        lib.ob_stream_profile_list_get_video_stream_profile.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            err_ptr,
        ]
        lib.ob_stream_profile_list_get_video_stream_profile.restype = ctypes.c_void_p

        lib.ob_stream_profile_list_count.argtypes = [ctypes.c_void_p, err_ptr]
        lib.ob_stream_profile_list_count.restype = ctypes.c_uint32

        lib.ob_delete_stream_profile_list.argtypes = [ctypes.c_void_p, err_ptr]
        lib.ob_delete_stream_profile_list.restype = None
        lib.ob_delete_stream_profile.argtypes = [ctypes.c_void_p, err_ptr]
        lib.ob_delete_stream_profile.restype = None

        lib.ob_config_enable_stream.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            err_ptr,
        ]
        lib.ob_config_enable_stream.restype = None

        lib.ob_config_set_align_mode.argtypes = [ctypes.c_void_p, ctypes.c_int, err_ptr]
        lib.ob_config_set_align_mode.restype = None

        lib.ob_get_d2c_depth_profile_list.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            err_ptr,
        ]
        lib.ob_get_d2c_depth_profile_list.restype = ctypes.c_void_p

        lib.ob_pipeline_get_device.argtypes = [ctypes.c_void_p, err_ptr]
        lib.ob_pipeline_get_device.restype = ctypes.c_void_p
        lib.ob_delete_device.argtypes = [ctypes.c_void_p, err_ptr]
        lib.ob_delete_device.restype = None

        lib.ob_pipeline_start_with_config.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            err_ptr,
        ]
        lib.ob_pipeline_start_with_config.restype = None
        lib.ob_pipeline_stop.argtypes = [ctypes.c_void_p, err_ptr]
        lib.ob_pipeline_stop.restype = None

        lib.ob_create_pointcloud_filter.argtypes = [err_ptr]
        lib.ob_create_pointcloud_filter.restype = ctypes.c_void_p
        lib.ob_pointcloud_filter_set_camera_param.argtypes = [
            ctypes.c_void_p,
            OBCameraParam,
            err_ptr,
        ]
        lib.ob_pointcloud_filter_set_camera_param.restype = None
        lib.ob_pointcloud_filter_set_point_format.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            err_ptr,
        ]
        lib.ob_pointcloud_filter_set_point_format.restype = None
        lib.ob_pointcloud_filter_set_position_data_scale.argtypes = [
            ctypes.c_void_p,
            ctypes.c_float,
            err_ptr,
        ]
        lib.ob_pointcloud_filter_set_position_data_scale.restype = None
        lib.ob_delete_filter.argtypes = [ctypes.c_void_p, err_ptr]
        lib.ob_delete_filter.restype = None

        lib.ob_filter_process.argtypes = [ctypes.c_void_p, ctypes.c_void_p, err_ptr]
        lib.ob_filter_process.restype = ctypes.c_void_p

        lib.ob_pipeline_get_camera_param.argtypes = [ctypes.c_void_p, err_ptr]
        lib.ob_pipeline_get_camera_param.restype = OBCameraParam

        lib.ob_pipeline_wait_for_frameset.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            err_ptr,
        ]
        lib.ob_pipeline_wait_for_frameset.restype = ctypes.c_void_p

        lib.ob_frameset_depth_frame.argtypes = [ctypes.c_void_p, err_ptr]
        lib.ob_frameset_depth_frame.restype = ctypes.c_void_p

        lib.ob_depth_frame_get_value_scale.argtypes = [ctypes.c_void_p, err_ptr]
        lib.ob_depth_frame_get_value_scale.restype = ctypes.c_float

        lib.ob_delete_frame.argtypes = [ctypes.c_void_p, err_ptr]
        lib.ob_delete_frame.restype = None

        lib.ob_frame_data_size.argtypes = [ctypes.c_void_p, err_ptr]
        lib.ob_frame_data_size.restype = ctypes.c_uint32

        lib.ob_frame_data.argtypes = [ctypes.c_void_p, err_ptr]
        lib.ob_frame_data.restype = ctypes.c_void_p

    # ------------------------------------------------------------------
    # Error-aware call helper.
    # ------------------------------------------------------------------
    def call(self, func, *args, ignore_error: bool = False):
        error = ErrorHandle()
        result = func(*args, ctypes.byref(error))
        if error.value:
            err_handle = ErrorHandle(error.value)
            if ignore_error:
                self._lib.ob_delete_error(err_handle)
                return None
            message = self._lib.ob_error_message(err_handle)
            func_name = self._lib.ob_error_function(err_handle)
            arg_list = self._lib.ob_error_args(err_handle)
            exc_type = self._lib.ob_error_exception_type(err_handle)
            self._lib.ob_delete_error(err_handle)
            raise OrbbecError(
                f"{func_name.decode() if func_name else func.__name__}: "
                f"{message.decode() if message else 'Unknown error'} "
                f"(args={arg_list.decode() if arg_list else ''}, type={exc_type})"
            )
        return result

    # ------------------------------------------------------------------
    # Convenience wrappers for readability.
    # ------------------------------------------------------------------
    @property
    def lib(self) -> ctypes.CDLL:
        return self._lib


# ---------------------------------------------------------------------------
# PLY writers.
# ---------------------------------------------------------------------------
def _rotate_axes(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Mantiene X, invierte Y y Z para orientación correcta sin espejo."""
    # Mantiene X para evitar el efecto espejo
    # Invierte Y para que no esté de cabeza
    # Invierte Z para que esté de frente
    return x, -y, -z


def _downsample_points(
    points: Sequence[Tuple[float, ...]], voxel_size: float
) -> List[Tuple[float, ...]]:
    """Simple voxel grid downsampling for tuples containing XYZ (+ optional extra data)."""

    if voxel_size <= 0:
        return list(points)

    inverse = 1.0 / voxel_size
    voxels: dict[Tuple[int, int, int], Tuple[float, ...]] = {}
    for point in points:
        x, y, z = point[:3]
        key = (
            int(math.floor(x * inverse)),
            int(math.floor(y * inverse)),
            int(math.floor(z * inverse)),
        )
        # Keep the first point that lands in a voxel to preserve overall structure
        voxels.setdefault(key, point)
    return list(voxels.values())


def save_points_to_ply(
    sdk: OrbbecSDK, frame: ctypes.c_void_p, path: Path, voxel_size: float
) -> None:
    data_size = sdk.call(sdk.lib.ob_frame_data_size, frame)
    count = data_size // ctypes.sizeof(OBPoint)
    data_ptr = sdk.call(sdk.lib.ob_frame_data, frame)
    point_array = ctypes.cast(data_ptr, ctypes.POINTER(OBPoint))
    rotated_points = [
        _rotate_axes(point_array[idx].x, point_array[idx].y, point_array[idx].z)
        for idx in range(count)
    ]
    filtered_points = _downsample_points(rotated_points, voxel_size)
    with path.open("w", encoding="utf-8") as ply:
        ply.write("ply\nformat ascii 1.0\n")
        ply.write(f"element vertex {len(filtered_points)}\n")
        ply.write("property float x\nproperty float y\nproperty float z\n")
        ply.write("end_header\n")
        for x, y, z in filtered_points:
            ply.write(f"{x:.3f} {y:.3f} {z:.3f}\n")


def save_rgb_points_to_ply(
    sdk: OrbbecSDK, frame: ctypes.c_void_p, path: Path, voxel_size: float
) -> None:
    data_size = sdk.call(sdk.lib.ob_frame_data_size, frame)
    count = data_size // ctypes.sizeof(OBColorPoint)
    data_ptr = sdk.call(sdk.lib.ob_frame_data, frame)
    point_array = ctypes.cast(data_ptr, ctypes.POINTER(OBColorPoint))
    rotated_points = [
        _rotate_axes(point_array[idx].x, point_array[idx].y, point_array[idx].z)
        + (
            int(point_array[idx].r),
            int(point_array[idx].g),
            int(point_array[idx].b),
        )
        for idx in range(count)
    ]
    filtered_points = _downsample_points(rotated_points, voxel_size)
    with path.open("w", encoding="utf-8") as ply:
        ply.write("ply\nformat ascii 1.0\n")
        ply.write(f"element vertex {len(filtered_points)}\n")
        ply.write(
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        )
        ply.write("end_header\n")
        for x, y, z, r, g, b in filtered_points:
            ply.write(f"{x:.3f} {y:.3f} {z:.3f} {r} {g} {b}\n")


# ---------------------------------------------------------------------------
# Pipeline configuration helpers.
# ---------------------------------------------------------------------------
def configure_streams(sdk: OrbbecSDK, pipeline: ctypes.c_void_p, config: ctypes.c_void_p):
    lib = sdk.lib
    color_profiles = sdk.call(
        lib.ob_pipeline_get_stream_profile_list,
        pipeline,
        OB_SENSOR_COLOR,
        ignore_error=True,
    )
    color_profile = None
    align_mode = ALIGN_DISABLE

    if color_profiles:
        color_profile = sdk.call(
            lib.ob_stream_profile_list_get_profile,
            color_profiles,
            OB_PROFILE_DEFAULT,
        )
        if color_profile:
            sdk.call(lib.ob_config_enable_stream, config, color_profile)

    depth_profiles = None
    if color_profile:
        for candidate_mode in (ALIGN_D2C_HW_MODE, ALIGN_D2C_SW_MODE):
            depth_profiles = sdk.call(
                lib.ob_get_d2c_depth_profile_list,
                pipeline,
                color_profile,
                candidate_mode,
                ignore_error=True,
            )
            if depth_profiles:
                count = sdk.call(lib.ob_stream_profile_list_count, depth_profiles)
                if count > 0:
                    align_mode = candidate_mode
                    break
                sdk.call(lib.ob_delete_stream_profile_list, depth_profiles)
                depth_profiles = None
        if not depth_profiles:
            depth_profiles = sdk.call(
                lib.ob_pipeline_get_stream_profile_list,
                pipeline,
                OB_SENSOR_DEPTH,
            )
    else:
        depth_profiles = sdk.call(
            lib.ob_pipeline_get_stream_profile_list,
            pipeline,
            OB_SENSOR_DEPTH,
        )

    depth_profile = None
    if depth_profiles:
        if color_profile:
            color_fps = sdk.call(lib.ob_video_stream_profile_fps, color_profile)
            depth_profile = sdk.call(
                lib.ob_stream_profile_list_get_video_stream_profile,
                depth_profiles,
                0,
                0,
                0,
                color_fps,
                ignore_error=True,
            )
        if not depth_profile:
            depth_profile = sdk.call(
                lib.ob_stream_profile_list_get_profile,
                depth_profiles,
                OB_PROFILE_DEFAULT,
            )
        sdk.call(lib.ob_config_enable_stream, config, depth_profile)
        sdk.call(lib.ob_config_set_align_mode, config, align_mode)

    return color_profiles, color_profile, depth_profiles, depth_profile, align_mode


# ---------------------------------------------------------------------------
# Point-cloud capture logic.
# ---------------------------------------------------------------------------
def capture_pointcloud(
    sdk: OrbbecSDK,
    pipeline: ctypes.c_void_p,
    point_filter: ctypes.c_void_p,
    timeout_ms: int,
    max_attempts: int,
    point_format: int,
    destination: Path,
    voxel_size: float,
) -> bool:
    attempts = 0
    lib = sdk.lib
    while attempts < max_attempts:
        frameset = sdk.call(lib.ob_pipeline_wait_for_frameset, pipeline, timeout_ms)
        if not frameset:
            attempts += 1
            continue
        depth_frame = sdk.call(lib.ob_frameset_depth_frame, frameset)
        if not depth_frame:
            sdk.call(lib.ob_delete_frame, frameset)
            attempts += 1
            continue
        depth_scale = sdk.call(lib.ob_depth_frame_get_value_scale, depth_frame)
        sdk.call(lib.ob_delete_frame, depth_frame)
        sdk.call(
            lib.ob_pointcloud_filter_set_position_data_scale,
            point_filter,
            depth_scale,
        )
        sdk.call(lib.ob_pointcloud_filter_set_point_format, point_filter, point_format)
        points_frame = sdk.call(lib.ob_filter_process, point_filter, frameset)
        sdk.call(lib.ob_delete_frame, frameset)
        if points_frame:
            if point_format == OB_FORMAT_RGB_POINT:
                save_rgb_points_to_ply(sdk, points_frame, destination, voxel_size)
            else:
                save_points_to_ply(sdk, points_frame, destination, voxel_size)
            sdk.call(lib.ob_delete_frame, points_frame)
            return True
        attempts += 1
    return False


# ---------------------------------------------------------------------------
# Command-line interface.
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["rgbd", "depth", "both"],
        default="both",
        help="Select which type of point cloud to save.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_MS,
        help="Frame wait timeout in milliseconds (default: 100).",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=10,
        help="Maximum number of frame requests per capture (default: 10).",
    )
    parser.add_argument(
        "--capture-id",
        type=int,
        default=DEFAULT_SILO_ID,
        help="Identifier to prefix generated filenames with (e.g., silo index).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the PLY files will be written.",
    )
    parser.add_argument(
        "--library-path",
        type=Path,
        default=DEFAULT_LIBRARY_PATH,
        help="Optional explicit path to libOrbbecSDK (so/dll).",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=DEFAULT_VOXEL_SIZE,
        help=(
            "Size of the voxel grid (in meters) used to downsample the exported point cloud. "
            "Set to 0 to keep every point."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sdk = OrbbecSDK(args.library_path)
    sdk.call(sdk.lib.ob_set_logger_severity, OB_LOG_SEVERITY_ERROR)

    pipeline = sdk.call(sdk.lib.ob_create_pipeline)
    config = sdk.call(sdk.lib.ob_create_config)

    color_profiles = color_profile = None
    depth_profiles = depth_profile = None
    device = None
    point_filter = None

    try:
        (
            color_profiles,
            color_profile,
            depth_profiles,
            depth_profile,
            _align_mode,
        ) = configure_streams(sdk, pipeline, config)

        device = sdk.call(sdk.lib.ob_pipeline_get_device, pipeline)
        sdk.call(sdk.lib.ob_pipeline_start_with_config, pipeline, config)

        point_filter = sdk.call(sdk.lib.ob_create_pointcloud_filter)
        camera_param = sdk.call(sdk.lib.ob_pipeline_get_camera_param, pipeline)
        sdk.call(
            sdk.lib.ob_pointcloud_filter_set_camera_param,
            point_filter,
            camera_param,
        )

        modes: List[str]
        if args.mode == "both":
            modes = ["rgbd", "depth"]
        else:
            modes = [args.mode]

        for mode in modes:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            suffix = "_rgb" if mode == "rgbd" else ""
            filename = f"{args.capture_id}_{timestamp}{suffix}.ply"
            destination = args.output_dir / filename
            point_format = OB_FORMAT_RGB_POINT if mode == "rgbd" else OB_FORMAT_POINT
            print(f"Capturing {mode.upper()} point cloud to {destination} ...")
            ok = capture_pointcloud(
                sdk,
                pipeline,
                point_filter,
                args.timeout,
                args.max_attempts,
                point_format,
                destination,
                args.voxel_size,
            )
            if not ok:
                raise OrbbecError(
                    f"Unable to create a {mode} point cloud after {args.max_attempts} attempts."
                )
            print(f"Saved {destination}")

        print("Done. Press Ctrl+C to quit or run again for new captures.")
        return 0

    finally:
        lib = sdk.lib
        if point_filter:
            sdk.call(lib.ob_delete_filter, point_filter, ignore_error=True)
        sdk.call(lib.ob_pipeline_stop, pipeline, ignore_error=True)
        if device:
            sdk.call(lib.ob_delete_device, device, ignore_error=True)
        if config:
            sdk.call(lib.ob_delete_config, config, ignore_error=True)
        if pipeline:
            sdk.call(lib.ob_delete_pipeline, pipeline, ignore_error=True)
        if color_profile:
            sdk.call(lib.ob_delete_stream_profile, color_profile, ignore_error=True)
        if depth_profile:
            sdk.call(lib.ob_delete_stream_profile, depth_profile, ignore_error=True)
        if color_profiles:
            sdk.call(lib.ob_delete_stream_profile_list, color_profiles, ignore_error=True)
        if depth_profiles:
            sdk.call(lib.ob_delete_stream_profile_list, depth_profiles, ignore_error=True)


if __name__ == "__main__":
    raise SystemExit(main())