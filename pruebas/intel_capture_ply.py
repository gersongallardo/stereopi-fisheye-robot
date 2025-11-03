import pyrealsense2 as rs
import logging
from logging.handlers import RotatingFileHandler
import time, os
from datetime import datetime
import cv2
import numpy as np
import re

image_rgb = True
image_ir = True
output_dir = "ply_pablo"
os.makedirs(output_dir, exist_ok=True)

log_directory = os.path.join(os.getcwd(), 'logs')
os.makedirs(log_directory, exist_ok=True)
log_filename = os.path.join(log_directory, f'silo_register_{datetime.now().strftime("%Y%m%d")}.log')

logger = logging.getLogger('silo_register')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

file_handler = RotatingFileHandler(log_filename, maxBytes=5*1024*1024, backupCount=10, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Configurar los flujos
logger.info('Configurando flujos de profundidad y color...')
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
if image_rgb:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
if image_ir:
    config.enable_stream(rs.stream.infrared, 640, 480, rs.format.y8, 30)

context = rs.context()
max_retries = 3
retry_delay = 5

for attempt in range(max_retries):
    devices = context.query_devices()
    if len(devices) > 0:
        logger.info("Dispositivo RealSense detectado")
        try:
            pipeline.start(config)
            logger.info("Transmisión iniciada")
            break
        except RuntimeError as e:
            logger.error(f"Error al iniciar: {str(e)}")
            for dev in context.devices:
                dev.hardware_reset()
            time.sleep(retry_delay)
    else:
        logger.warning("No se detectó ningún dispositivo, reiniciando...")
        for dev in context.devices:
            dev.hardware_reset()
        time.sleep(retry_delay)
else:
    logger.error("No se pudo iniciar el dispositivo. Reiniciando sistema...")
    time.sleep(2)
    os.system("sudo reboot")

# Configurar sensor
device = pipeline.get_active_profile().get_device()
depth_sensor = device.first_depth_sensor()
new_power = depth_sensor.get_option_range(rs.option.laser_power).max
depth_sensor.set_option(rs.option.laser_power, new_power)
logger.info(f"Potencia láser establecida en: {new_power}")

# Inicializar contador para los archivos
# Buscar el último número usado en los archivos existentes
existing_files = [f for f in os.listdir(output_dir) if f.endswith('.ply')]
ply_counter = 0
pattern = re.compile(r'captura_(\d+)\.ply')

for f in existing_files:
    match = pattern.match(f)
    if match:
        num = int(match.group(1))
        if num >= ply_counter:
            ply_counter = num + 1

logger.info("Iniciando captura continua. Presiona 's' para guardar un .ply, 'q' para salir.")

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame() if image_rgb else None
        ir_frame = frames.get_infrared_frame() if image_ir else None

        # Mostrar una vista previa si se quiere (opcional)
        if image_rgb and color_frame:
            color_image = np.asanyarray(color_frame.get_data())
            cv2.imshow('Color Frame', color_image)

        if image_ir and ir_frame:
            ir_image = np.asanyarray(ir_frame.get_data())
            cv2.imshow('IR Frame', ir_image)

        if depth_frame:
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            cv2.imshow('Depth Frame', depth_colormap)

        key = cv2.waitKey(1)

        if key == ord('s'):
            # Guardar PLY
            ply_filename = os.path.join(output_dir, f"captura_{ply_counter:03d}.ply")
            ply = rs.save_to_ply(ply_filename)
            ply.set_option(rs.save_to_ply.option_ply_binary, True)
            ply.set_option(rs.save_to_ply.option_ignore_color, True)
            ply.set_option(rs.save_to_ply.option_ply_mesh, True)
            ply.set_option(rs.save_to_ply.option_ply_normals, False)

            ply.process(frames)
            logger.info(f"Archivo guardado: {ply_filename}")
            print(f"PLY guardado: {ply_filename}")
            ply_counter += 1

        elif key == ord('q'):
            logger.info("Saliendo del programa por orden del usuario")
            break

except Exception as e:
    logger.error(f"Error durante la captura: {str(e)}")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    logger.info("Pipeline detenido. Programa finalizado.")
