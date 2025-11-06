import open3d as o3d
import os
import inspect
import silos_objects as silo_module
from silos_objects import Silo
import utils as utils


# Directorios
input_ply_directory = "/home/nicolas.donoso/silo_files/data"
output_ply_directory = "/home/nicolas.donoso/silo_files/view"
silos_mesh_directory = "/home/nicolas.donoso/"

# Configuración del logger
logger = utils.setup_logging(log_dir="logs/ply_filter", log_function="ply_filter")

# Extraer todas las instancias de Silo definidas en el módulo
silos = [
    value for name, value in inspect.getmembers(silo_module)
    if isinstance(value, Silo)
]

# Procesar cada silo automáticamente
for silo in silos:
    logger.info(f"Procesando silo: {silo.empresa}-{silo.ubicacion}_{silo.silo}_{silo.sensor}")

    try:
        # Extraer el último archivo PLY
        filename, file_path = utils.last_file_ply(silo, input_ply_directory)
        logger.info(f"Archivo seleccionado: {filename}")

        # Filtrar el archivo PLY
        filtered_ply = utils.load_and_filter_ply(silo, filename, file_path, silos_mesh_directory)
        logger.info(f"Archivo filtrado: {filename}")

        # Guardar el archivo filtrado
        output_filename = f'{silo.empresa}-{silo.ubicacion}_{silo.silo}_{silo.sensor}.ply'
        output_path = os.path.join(output_ply_directory, output_filename)
            # Verificar que el directorio de salida existe
        if not os.path.isdir(output_ply_directory):
            raise FileNotFoundError(f"Directorio de salida no encontrado: {output_ply_directory}")

        success = o3d.io.write_point_cloud(output_path, filtered_ply, compressed=True)
        if not success:
            raise IOError(f"No se pudo guardar el archivo PLY en {output_path}")
        logger.info(f"Guardado en: {output_path}")

    except FileNotFoundError as fnf_error:
        logger.error(f"Archivo no encontrado: {fnf_error}")
        continue
    except IOError as ioe:
        logger.error(f"Error de escritura al guardar archivo: {ioe}")
        continue
    except ValueError as ve:
        logger.error(f"Valor incorrecto al procesar {silo.empresa}-{silo.ubicacion}_{silo.silo}_{silo.sensor}: {ve}")
        continue
    except Exception as e:
        logger.error(f"Error inesperado al procesar {silo.empresa}-{silo.ubicacion}_{silo.silo}_{silo.sensor}: {e}")
        continue
    logger.info("")