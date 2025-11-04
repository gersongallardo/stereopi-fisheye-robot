import os
from datetime import datetime

import cv2
import depthai as dai
import numpy as np
import open3d as o3d


DOT_PROJECTOR_INTENSITY = 1
FLOOD_LIGHT_INTENSITY = 0

NODE_ID = int(os.getenv("NODE", 0))
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY_EXPORT", ".")
FRAME_TARGET = 50
FPS = 5


def configure_pipeline() -> dai.Pipeline:
    pipeline = dai.Pipeline()

    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)
    depth = pipeline.create(dai.node.StereoDepth)
    pointcloud = pipeline.create(dai.node.PointCloud)

    xout_pointcloud = pipeline.create(dai.node.XLinkOut)
    xout_pointcloud.setStreamName("pcl")

    xout_mono_right = pipeline.create(dai.node.XLinkOut)
    xout_mono_right.setStreamName("mono_right")

    mono_resolution = dai.MonoCameraProperties.SensorResolution.THE_480_P

    for mono, socket in ((mono_left, "left"), (mono_right, "right")):
        mono.setResolution(mono_resolution)
        mono.setCamera(socket)
        mono.setFps(FPS)

    depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
    depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_5x5)
    depth.setLeftRightCheck(True)
    depth.setSubpixel(True)

    resolution_dimensions = {
        dai.MonoCameraProperties.SensorResolution.THE_400_P: (640, 400),
        dai.MonoCameraProperties.SensorResolution.THE_480_P: (640, 480),
        dai.MonoCameraProperties.SensorResolution.THE_720_P: (1280, 720),
        dai.MonoCameraProperties.SensorResolution.THE_800_P: (1280, 800),
    }
    width, height = resolution_dimensions[mono_resolution]
    depth.setOutputSize(width, height)
    depth.setDepthAlign(dai.CameraBoardSocket.LEFT)

    mono_left.out.link(depth.left)
    mono_right.out.link(depth.right)
    depth.depth.link(pointcloud.inputDepth)
    pointcloud.outputPointCloud.link(xout_pointcloud.input)
    mono_right.out.link(xout_mono_right.input)

    return pipeline


def stabilize(q_pcl: dai.DataOutputQueue, q_ir: dai.DataOutputQueue) -> None:
    for _ in range(FRAME_TARGET):
        q_pcl.get()
        q_ir.get()


def capture_pointcloud(q_pcl: dai.DataOutputQueue, filename: str) -> None:
    in_pcl = q_pcl.get()
    points = in_pcl.getPoints().astype(np.float64) / 1000.0
    points[:, 2] *= -1

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    ply_path = os.path.join(DATA_DIRECTORY, f"{filename}.ply")
    o3d.io.write_point_cloud(ply_path, pcd, write_ascii=False)
    print(f"PLY guardado: {ply_path}")


def capture_ir(q_ir: dai.DataOutputQueue, filename: str) -> None:
    frame = q_ir.get().getCvFrame()
    ir_path = os.path.join(DATA_DIRECTORY, f"{filename}_ir.jpg")
    cv2.imwrite(ir_path, frame)
    print(f"Imagen IR guardada: {ir_path}")


def main() -> None:
    pipeline = configure_pipeline()

    with dai.Device(pipeline) as device:
        device.setIrLaserDotProjectorIntensity(DOT_PROJECTOR_INTENSITY)
        device.setIrFloodLightIntensity(FLOOD_LIGHT_INTENSITY)

        q_pcl = device.getOutputQueue(name="pcl", maxSize=1, blocking=True)
        q_ir = device.getOutputQueue(name="mono_right", maxSize=1, blocking=True)

        stabilize(q_pcl, q_ir)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{NODE_ID}_{timestamp}"

        capture_pointcloud(q_pcl, filename)
        capture_ir(q_ir, filename)


if __name__ == "__main__":
    os.makedirs(DATA_DIRECTORY, exist_ok=True)
    main()
