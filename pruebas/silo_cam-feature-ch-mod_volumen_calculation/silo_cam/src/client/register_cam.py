import pyrealsense2 as rs
from sensor_adquicion import SensorCollector
from exporter_data import ExporterData
import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
import time
import numpy as np


SENSOR_AVAILABLE = False

try:
    import board
    SENSOR_AVAILABLE = True
except ImportError:
    pass 

# Crear el directorio de logs si no existe
log_directory = os.path.join(os.getcwd(), 'logs')
os.makedirs(log_directory, exist_ok=True)

# Nombre del archivo de log con fecha
log_filename = os.path.join(log_directory, f'silo_register_{datetime.now().strftime("%Y%m%d")}.log')

# Configurar el logger
logger = logging.getLogger('silo_register')
logger.setLevel(logging.DEBUG)

# Formato del log
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Handler para archivo con rotación (máximo 5MB por archivo, máximo 10 archivos)
file_handler = RotatingFileHandler(
    log_filename,
    maxBytes=5*1024*1024,  # 5MB
    backupCount=10,
    encoding='utf-8'
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

# Handler para la terminal
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Agregar los handlers al logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Configuración
logger.info('Importando variables...')
node = int(os.getenv('NODE', 0))
data_directory = os.getenv('DATA_DIRECTORY_EXPORT', '/home/pi/Documents') 
csv_name = os.getenv('CSV_NAME', 'data')  
image_infra = os.getenv('IMAGE_IR', 'False').lower() == 'true' 

# Inicializar export data
export_dat = ExporterData(data_directory, csv_name, node, logger=logger)

if SENSOR_AVAILABLE:
    # Crear instancia del colector
    collector = SensorCollector(logger)
    # Agregar sensores
    collector.add_dht22_sensor('indoor_sensor', board.D20)  # DHT22 en GPIO 20
    collector.add_digital_sensor('AC_OK', board.D21)  # Entrada digital en GPIO 21

# Configurar los flujos de profundidad e infrarrojo de la cámara
logger.info('Configurando flujos de profundidad e infrarrojo...')
pipeline = rs.pipeline()
config = rs.config()


config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
if image_infra:
    config.enable_stream(rs.stream.infrared, 640, 480, rs.format.y8, 30)

context = rs.context()
max_retries = 3  # Número máximo de intentos
retry_delay = 5  # Segundos de espera entre intentos

for attempt in range(max_retries):
    devices = context.query_devices()  # Verifica si hay dispositivos conectados

    if len(devices) > 0:
        logger.info("Dispositivo RealSense detectado, intentando iniciar transmisión...")

        try:
            pipeline.start(config)
            logger.info("Transmisión iniciada exitosamente")
            break  # Sale del bucle si todo funciona
        
        except RuntimeError as e:
            logger.error(f"Error al iniciar la transmisión: {str(e)}")
            logger.warning(f"Intentando reinicio de hardware (Intento {attempt + 1}/{max_retries})...")

            # Reiniciar hardware RealSense
            for dev in context.devices:
                dev.hardware_reset()

            time.sleep(retry_delay)  # Espera para que el dispositivo se reinicie

    else:
        logger.warning(f"No se detectó ningún dispositivo RealSense. Intentando reinicio de hardware (Intento {attempt + 1}/{max_retries})...")
        
        # Reiniciar hardware RealSense
        for dev in context.devices:
            dev.hardware_reset()

        time.sleep(retry_delay)  # Espera para que el dispositivo se reinicie

if (attempt + 1) >= max_retries:
    logger.error("No se pudo conectar con el dispositivo RealSense después de varios intentos. Reiniciando la Raspberry Pi...")
    time.sleep(2)  # Pequeña pausa antes de reiniciar
    os.system("sudo reboot")



# Obtener el dispositivo desde el pipeline
device = pipeline.get_active_profile().get_device()

# Obtener el sensor de profundidad
depth_sensor = device.first_depth_sensor()

# Configurar potencia de laser en max 360 mW
depth_sensor.set_option(rs.option.laser_power, 360)

# Contador de fotogramas
n_frames = 0
frames = None
n_samples = 10
samples = []

try:
    for sample in range(n_samples):  # Toma n capturas
        logger.info(f'Iniciando captura número {sample+1}/{n_samples}...')
        for attempts in range(max_retries): # Intentar 3 veces
            try:
                logger.info(f'Intento de captura {attempts+1}/3')
                # Esperar hasta alcanzar 90 fotogramas
                logger.info('Esperando 90 fotogramas...')
                while n_frames < 90:
                    pipeline.wait_for_frames()
                    n_frames += 1

                # Capturar fotograma después de 90 iteraciones
                frames = pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                
                logger.info('Captura exitosa')
                success = True
                # Reiniciar el contador de fotogramas
                n_frames = 0
                break 
            
            
            except Exception as e:
                n_frames = 0
                logger.error(f'Error en el intento: {str(e)}')
                logger.warning(f"Intentando reinicio de hardware (Intento {attempts + 1}/{max_retries})...")

                # Reiniciar hardware RealSense
                for devs in context.devices:
                    devs.hardware_reset()

                time.sleep(retry_delay)  # Espera para que el dispositivo se reinicie
                
                if (attempts +1) >= max_retries:
                    logger.error("Número máximo de intentos alcanzados.")
                    raise e
                
        # Recolectar muestra
        if frames is not None:
            samples.append(depth_frame)
            logger.info(f'Matriz recolectada {sample+1}/{n_samples}')
            
        # Última muestra
        if sample +1 >= n_samples:
            # Almacenar ultima imagen ir
            ir_frame = frames.get_infrared_frame() if image_infra else None
            
            if SENSOR_AVAILABLE:
                # Adquisicion sensores
                data = collector.collect_data()
                logger.info(f'Datos recolectados: {data}')
                # Guardar datos en csv
                export_dat.save_data_sensors(data)
        
                    
except Exception as e:
    logger.error(f'No se pudo capturar ninguna muestra. Error en la captura: {str(e)}')
    logger.warning("Se reiniciará la Raspberry Pi...")
    time.sleep(2)  # Espera final
    os.system("sudo reboot")  # Si sigue fallando, reinicia la Raspberry Pi

# Detener la transmisión y liberar recursos
pipeline.stop()
collector.cleanup()
logger.info('Transmisión finalizada.')


# Obtener los arrays de profundidad
samples_arrays = [np.array(frame.get_data()) for frame in samples]

# Promediar los valores de profundidad
depth_mean = np.nanmean(samples_arrays, axis=0)

# Convertir a uint16 antes de guardar (descarta decimales)
depth_mean = depth_mean.astype(np.uint16)

# Guardar matriz de profundidad e infrarrojo
export_dat.save_matrix(depth_mean, ir_frame)
logger.info('Programa finalizado.') 
logger.info('') 