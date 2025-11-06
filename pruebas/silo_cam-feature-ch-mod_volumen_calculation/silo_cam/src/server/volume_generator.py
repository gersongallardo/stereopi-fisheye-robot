import volume_calculation_of_ply_file as vc
import pyvista as pv
import os, sys
import inspect
import logging
import requests
import calendar
import pytz
from datetime import datetime
import silos_objects as silo_module
from silos_objects import Silo



# Configuración del logger
def setup_logging(log_dir="logs"):
    """Configura el sistema de logging"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = os.path.join(log_dir, f"send_cacheton_{datetime.now().strftime('%Y%m%d')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger('send_cacheton')
    logger.setLevel(logging.INFO)
    return logger


def arreglo_de_malla_procesada(malla, nombre_ply, silo_object, logger):
    try:
        logger.info(f"Procesando malla.")
        malla_procesada, z_mean = vc.proccesing_mesh_with_z_mean(malla, silo_object)
    except Exception as e:
        logger.error(f"Error al procesar la malla: {e}")
    return malla_procesada, nombre_ply, z_mean


def get_results_of_volume_meassurement(malla_procesada, z_mean, silo_object,logger):
    # Obtener los valores
    try:
        # Calculo de volumen según forma de silo
        logger.info(f"Estimando volumen de la malla.")
        if silo_object.alto_prisma == None:
            volumen_silo = vc.volume_measurement_of_silo(malla_procesada, z_mean, silo_object)
        else:
            volumen_silo = vc.volume_measurement_of_silo_with_prism(malla_procesada, z_mean, silo_object)

    except Exception as e:
        logger.error(f"Error al estimar volumen de la malla")
        volumen_silo = 0

    return volumen_silo



def obtener_ultimo_archivo(carpeta, logger):
    archivos = [
        os.path.join(carpeta, f)
        for f in os.listdir(carpeta)
        if f.endswith(".ply")
    ]

    def extraer_fecha_hora(nombre_archivo):
        nombre = os.path.basename(nombre_archivo)
        try:
            partes = nombre.split("_")
            fecha_str = partes[1]
            hora_str = partes[2].replace(".ply", "")
            return int(calendar.timegm(datetime.strptime(f"{fecha_str} {hora_str}", "%Y-%m-%d %H:%M:%S").timetuple()))
        except Exception as e:
            logger.error(f"No se pudo extraer fecha y hora de {nombre_archivo}: {e}")
            return datetime.min

    if not archivos:
        return None, None

    archivos.sort(key=extraer_fecha_hora)
    ultimo_archivo = archivos[-1]
    return ultimo_archivo, os.path.basename(ultimo_archivo)


def calculo_de_valores(nombre_ply, volumen, nivel, silo_object):
    # Convertir nombres a fechas UNIX
    fecha_str = nombre_ply.split("_")[1] + "_" + nombre_ply.split("_")[2].replace(".ply", "")

    # Zona horaria de Chile (considera horario de verano si aplica)
    chile = pytz.timezone("Etc/GMT+4")
    fecha_chile = chile.localize(datetime.strptime(fecha_str, "%Y-%m-%d_%H:%M:%S"))
    # Convertir a UTC
    fecha_utc = fecha_chile.astimezone(pytz.utc)
    # Convertir a timestamp (Unix time)
    fecha_unix = int(calendar.timegm(fecha_utc.timetuple()))

    # Calcular porcentaje relativo
    min_vol = silo_object.volumen_minimo
    max_vol = silo_object.volumen_maximo
    porcentaje = round(((volumen - min_vol) / (max_vol- min_vol)) * 100, 1)
    if porcentaje > 100:
        porcentaje = 100.0

    # Calcular toneladas
    densidad_pellet = 0.734
    toneladas = round(volumen * densidad_pellet, 1)

    # Diccionario de valores
    status = 0
    resultado = {'unixtime': fecha_unix,
                 'volume': volumen,
                 'porcentage': porcentaje,
                 'toneladas': toneladas,
                 'level': nivel,
                 'status': status}

    return resultado

# Funciones para enviar a cacheton:

def send_data_to_server(server_url, silo_object, measurement, logger):
    """Envía los datos al servidor mediante una solicitud PUT"""
    try:
        url = f"{server_url}/{silo_object.empresa}-{silo_object.ubicacion}/slevel/{silo_object.silo}/{silo_object.sensor}/"

        # Parámetros
        params = {
            'time': measurement['unixtime'],
            'volume': measurement['volume'],
            'porcentage': measurement['porcentage'],
            'level': measurement['level'],
            'fill': measurement['toneladas'],
            'status': measurement['status']
        }

        logger.info(f"Enviando solicitud PUT a: {url} con parámetros: {params}")
        response = requests.put(url, params=params, timeout=30)  # Agregado timeout

        if response.status_code in [200, 201, 204]:
            logger.info(f"Datos enviados correctamente: {url} - Status: {response.status_code}")
            return True
        else:
            logger.error(f"Error HTTP al enviar datos: {url} - Status: {response.status_code} - Respuesta: {response.text}")
            return False

    except requests.exceptions.ConnectionError as ce:
        logger.error(f"Error de conexión al enviar datos: {url} - Error: {ce}")
        return False
    except requests.exceptions.Timeout as te:
        logger.error(f"Timeout al enviar datos: {url} - Error: {te}")
        return False
    except requests.exceptions.RequestException as re:
        logger.error(f"Error en la solicitud HTTP: {url} - Error: {re}")
        return False
    except Exception as e:
        logger.error(f"Excepción no controlada al enviar datos: {url} - Error: {str(e)}", exc_info=True)
        return False


def procesar_todos_los_silos(silo_module, directory_data, server_url, logger):
    silos = [
        value for name, value in inspect.getmembers(silo_module)
        if isinstance(value, Silo)
    ]

    for silo in silos:
        logger.info(f"Procesando silo: {silo.empresa}-{silo.ubicacion}_{silo.silo}_{silo.sensor}")

        ply_directory = f"{directory_data}/{silo.empresa}/{silo.ubicacion}/{silo.silo}_{silo.sensor}"

        if not os.path.isdir(ply_directory):
            logger.warning(f"Directorio no encontrado: {ply_directory}")
            continue

        ruta_completa, nombre_completo = obtener_ultimo_archivo(ply_directory, logger)

        if ruta_completa:
            try:
                malla = pv.read(ruta_completa)
            except Exception as e:
                logger.error(f"Error leyendo {ruta_completa}: {e}")
                continue

            malla_proc, nombre_valido, z_mean = arreglo_de_malla_procesada(malla, nombre_completo, silo, logger)

            if not malla_proc:
                logger.warning(f"Saltando archivo inválido: {nombre_completo}")
                continue

            volumen = get_results_of_volume_meassurement(malla_proc, z_mean, silo, logger)
            z_mean = round(z_mean + silo.alto_total, 2)

            try:
                valores = calculo_de_valores(nombre_valido, volumen, z_mean, silo)
                logger.info(f"Resultado para {nombre_completo}: {valores}")

            except Exception as e:
                logger.error(f"Error al hacer el cálculo de valores: {e}")
                continue

            ###__________________Enviar resultados a Cacheeton____________________###
            send_data_to_server(server_url, silo, valores, logger)

        else:
            logger.warning("No se encontró ningún archivo .ply en el directorio.")



def main():
    #  Directorio donde se encuentran los archivos .ply de todos los silos
    #directorio_data = "/home/innovex/Documentos/data_tenten/silo_files/data"
    directorio_data = "/home/nicolas.donoso/silo_files/data"

    # Dirección url
    sever_url = 'http://dataweb.innovex.cl:8888'

    logger = setup_logging(log_dir='logs/send_cacheton')

    # Procesar
    try:
        procesar_todos_los_silos(silo_module, directorio_data, sever_url, logger)

    except Exception as e:
        logger.error(f"Error al procesar el calculo o envío de datos: {e}")




if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)