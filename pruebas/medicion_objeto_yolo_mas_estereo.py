import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

#Use a pre-trained model
model = YOLO('yolov8n.pt')


pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start camera
pipeline.start(config)
align = rs.align(rs.stream.color)

#Function to get bounding boxes
def get_person_bboxes(frame):
    results = model(frame)
    person_bboxes = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id == 39:  # Clase 0 = persona en COCO
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_bboxes.append((x1, y1, x2, y2))

    return person_bboxes



#Function to get center of bounding boxes
def get_center_of_bboxes(person_bboxes):
    centers = []
    if len(person_bboxes) > 0:
        for box in person_bboxes:
            x1, y1, x2, y2 = box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            centers.append((cx, cy))
    return centers

#Function to get the z(c), that means the Z or depth value in the center of the bounding box.
def get_z_value_of_center(centers, depth_frame):
    z_of_centers = []
    if len(centers) > 0:
        for cx, cy in centers:
            z = depth_frame.get_distance(cx, cy)
            z_of_centers.append(z)
    return z_of_centers

#Function to get 2 2D points and depth value of the bounding box, top and bottom.
def get_top_and_bottom_points(person_bboxes, z_of_centers):
    top_points = []
    bottom_points = []

    if len(person_bboxes) > 0 and len(z_of_centers) > 0:
        for i, box in enumerate(person_bboxes):
            x1, y1, x2, y2 = box

            cx = (x1 + x2) // 2
            z = z_of_centers[i]

            top_points.append((cx, y1, z))
            bottom_points.append((cx, y2, z))

    return top_points, bottom_points

#Function to get 2 2D points and depth value of the bounding box, left and right.
def get_left_and_right_points(person_bboxes, z_of_centers):
    left_points = []
    right_points = []

    if len(person_bboxes) > 0 and len(z_of_centers) > 0:
        for i, box in enumerate(person_bboxes):
            x1, y1, x2, y2 = box

            cy = (y1 + y2) // 2
            z = z_of_centers[i]

            left_points.append((x1, cy, z))
            right_points.append((x2, cy, z))

    return left_points, right_points

#Function to get 3D points
def get_points_2D_in_3D(top_points_2D, bottom_points_2D, depth_frame):
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
    top_points_3D = []
    bottom_points_3D = []

    if len(top_points_2D) > 0 and len(bottom_points_2D) > 0:
        for top, bottom in zip(top_points_2D, bottom_points_2D):
            top_pixel = [top[0], top[1]]
            top_depth = top[2]
            top_3D = rs.rs2_deproject_pixel_to_point(depth_intrin, top_pixel, top_depth)
            top_points_3D.append(top_3D)

            bottom_pixel = [bottom[0], bottom[1]]
            bottom_depth = bottom[2]
            bottom_3D = rs.rs2_deproject_pixel_to_point(depth_intrin, bottom_pixel, bottom_depth)
            bottom_points_3D.append(bottom_3D)

    return top_points_3D, bottom_points_3D


#Function to meassure the euclidian distance between 2 3D points.
def get_distance(top_points_3D, bottom_points_3D):

    distances = []

    if len(top_points_3D) > 0 and len(bottom_points_3D) > 0:
        for top, bottom in zip(top_points_3D, bottom_points_3D):
            distance = np.linalg.norm(np.array(top) - np.array(bottom))
            distances.append(distance)

    return distances


try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        person_boxes = get_person_bboxes(color_image)

        center_of_boxes = get_center_of_bboxes(person_boxes)

        z_of_centers = get_z_value_of_center(center_of_boxes, depth_frame)

        top_points_2D, bottom_points_2D = get_top_and_bottom_points(person_boxes, z_of_centers)

        left_points_2D, right_points_2D = get_left_and_right_points(person_boxes, z_of_centers)

        top_points_3D, bottom_points_3D = get_points_2D_in_3D(top_points_2D, bottom_points_2D, depth_frame)

        left_points_3D, right_points_3D = get_points_2D_in_3D(left_points_2D, right_points_2D, depth_frame)

        heights = get_distance(top_points_3D, bottom_points_3D)

        wides = get_distance(left_points_3D, right_points_3D)

        for i, (x1, y1, x2, y2) in enumerate(person_boxes):
            cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            y_offset = y1 - 10

            if i < len(heights):
                height_text = f"Altura: {heights[i]:.2f} m"
                cv2.putText(color_image, height_text, (x1, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                y_offset -= 20


            if i < len(wides):
                width_text = f"Ancho: {wides[i]:.2f} m"
                cv2.putText(color_image, width_text, (x1, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 + RealSense", color_image)

        if cv2.waitKey(1) == 27:  # ESC to out
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()