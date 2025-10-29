import cv2
import depthai as dai
import numpy as np
import time

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)

xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutNN = pipeline.create(dai.node.XLinkOut)
xoutDepth = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")
xoutNN.setStreamName("detections")
xoutDepth.setStreamName("depth")

# Properties
camRgb.setPreviewSize(416, 416)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# Stereo depth configuration
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
stereo.setOutputSize(camRgb.getPreviewWidth(), camRgb.getPreviewHeight())

# Network specific settings
spatialDetectionNetwork.setBlobPath("yolov8n_coco_640x640.blob")
spatialDetectionNetwork.setConfidenceThreshold(0.5)
spatialDetectionNetwork.input.setBlocking(False)
spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
spatialDetectionNetwork.setDepthLowerThreshold(100)
spatialDetectionNetwork.setDepthUpperThreshold(5000)

# Yolo specific parameters
spatialDetectionNetwork.setNumClasses(80)
spatialDetectionNetwork.setCoordinateSize(4)
spatialDetectionNetwork.setAnchors([10,14, 23,27, 37,58, 81,82, 135,169, 344,319])
spatialDetectionNetwork.setAnchorMasks({"side26": [1,2,3], "side13": [3,4,5]})
spatialDetectionNetwork.setIouThreshold(0.5)

# Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

camRgb.preview.link(spatialDetectionNetwork.input)
spatialDetectionNetwork.passthrough.link(xoutRgb.input)
spatialDetectionNetwork.out.link(xoutNN.input)

stereo.depth.link(spatialDetectionNetwork.inputDepth)
spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)

# COCO labels
labelMap = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", 
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
            "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

def get_spatial_coordinates(detection, depth_frame, frame_shape):
    """
    Obtiene las coordenadas 3D de los puntos superior e inferior del bounding box
    """
    # Obtener dimensiones del frame
    height, width = frame_shape[:2]
    
    # Calcular coordenadas en píxeles
    x1 = int(detection.xmin * width)
    y1 = int(detection.ymin * height)
    x2 = int(detection.xmax * width)
    y2 = int(detection.ymax * height)
    
    # Centro horizontal del bounding box
    cx = (x1 + x2) // 2
    
    # Obtener profundidad en el centro del bbox
    # La profundidad espacial ya viene en milímetros
    depth_mm = detection.spatialCoordinates.z
    
    if depth_mm <= 0:
        return None, None
    
    # Conversión aproximada de coordenadas 2D a 3D usando la profundidad
    # Usando valores típicos de FOV de OAK-D
    # FOV horizontal ~69°, FOV vertical ~55°
    
    # Factor de conversión basado en la profundidad
    # Estos son valores aproximados para OAK-D
    x_factor = depth_mm / 1000.0  # Convertir a metros
    y_factor = depth_mm / 1000.0
    
    # Coordenadas 3D del punto superior (centrado en x)
    top_x = (cx - width/2) * x_factor * 0.0012  # Factor de escala aproximado
    top_y = (y1 - height/2) * y_factor * 0.0012
    top_z = depth_mm / 1000.0  # Convertir a metros
    
    # Coordenadas 3D del punto inferior (centrado en x)
    bottom_x = (cx - width/2) * x_factor * 0.0012
    bottom_y = (y2 - height/2) * y_factor * 0.0012
    bottom_z = depth_mm / 1000.0
    
    top_point = np.array([top_x, top_y, top_z])
    bottom_point = np.array([bottom_x, bottom_y, bottom_z])
    
    return top_point, bottom_point

def calculate_height(top_point, bottom_point):
    """
    Calcula la distancia euclidiana entre dos puntos 3D
    """
    if top_point is None or bottom_point is None:
        return None
    
    distance = np.linalg.norm(top_point - bottom_point)
    return distance

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    
    # Output queues
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
    qDepth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    
    frame = None
    detections = []
    startTime = time.monotonic()
    counter = 0
    color = (0, 255, 0)
    
    print("Presiona 'q' para salir")
    
    while True:
        inRgb = qRgb.get()
        inDet = qDet.get()
        inDepth = qDepth.get()
        
        if inRgb is not None:
            frame = inRgb.getCvFrame()
            
        if inDet is not None:
            detections = inDet.detections
            counter += 1
        
        if frame is not None:
            # Filtrar solo personas (clase 0 en COCO)
            person_detections = [d for d in detections if d.label == 0]
            
            for detection in person_detections:
                # Obtener bounding box
                bbox = [
                    int(detection.xmin * frame.shape[1]),
                    int(detection.ymin * frame.shape[0]),
                    int(detection.xmax * frame.shape[1]),
                    int(detection.ymax * frame.shape[0])
                ]
                
                # Calcular altura
                top_point, bottom_point = get_spatial_coordinates(detection, inDepth, frame.shape)
                height = calculate_height(top_point, bottom_point)
                
                # Dibujar bounding box
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                
                # Preparar texto
                y_offset = bbox[1] - 10
                
                # Mostrar altura
                if height is not None:
                    height_text = f"Altura: {height:.2f} m"
                    cv2.putText(frame, height_text, (bbox[0], y_offset),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    y_offset -= 20
                
                # Mostrar profundidad
                depth_text = f"Dist: {detection.spatialCoordinates.z/1000:.2f} m"
                cv2.putText(frame, depth_text, (bbox[0], y_offset),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                y_offset -= 20
                
                # Mostrar confianza
                conf_text = f"{int(detection.confidence * 100)}%"
                cv2.putText(frame, conf_text, (bbox[0], y_offset),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Mostrar FPS
            fps_text = f"FPS: {counter / (time.monotonic() - startTime):.2f}"
            cv2.putText(frame, fps_text, (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            cv2.imshow("OAK-D Person Height Detection", frame)
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    print(f"FPS final: {counter / (time.monotonic() - startTime):.2f}")

cv2.destroyAllWindows()