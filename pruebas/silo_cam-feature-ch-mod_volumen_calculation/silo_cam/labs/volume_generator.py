import volume_calculation_of_ply_file as vc
import pyvista as pv
import numpy as np
import sys
import inspect
import silos_objects as silo_module
from silos_objects import Silo
import utils as utils


def process_all_silos(silo_module, ply_directory, output_ply_directory, silos_mesh_directory, server_url, logger):
    silos = [
        value for name, value in inspect.getmembers(silo_module)
        if isinstance(value, Silo)
    ]

    for silo in silos:
        logger.info(f"Procesando silo: {silo.empresa}-{silo.ubicacion}_{silo.silo}_{silo.sensor}")

        # Obtener el último archivo PLY del silo
        filename, ply_path = utils.last_file_ply(silo, ply_directory)
        logger.info(f"Archivo seleccionado: {filename}")

        if ply_path:
            # Filtrado del ply
            try:
                logger.info(f"Filtrado de ply.")
                filter_ply = utils.filter_and_save_ply(filename, ply_path, output_ply_directory, silos_mesh_directory, logger, silo)
                mesh = pv.PolyData(np.asarray(filter_ply.points))
            except Exception as e:
                logger.error(f"Error filtrando o leyendo el ply: {e}")
                continue

            # Extruccion de la malla
            try:
                logger.info(f"Procesando malla.")
                processed_mesh, z_mean = utils.mesh_processing(mesh, filename, silo)
            except Exception as e:
                logger.error(f"Error al procesar la malla: {e}")
                continue

            # Estimación de volumen de la malla
            try:
                logger.info(f"Estimando volumen de la malla.")
                volume = utils.get_results_of_volume_meassurement(processed_mesh, z_mean, silo)
                z_mean = round(z_mean + silo.alto_total, 2)
            except Exception as e:
                logger.error(f"Error al estimar volumen de la malla: {e}")
                continue

            # Cálculo de valores
            try:
                measurements = utils.calculate_values(filename, volume, z_mean, silo)
                logger.info(f"Resultado para {filename}: {measurements}")

            except Exception as e:
                logger.error(f"Error al hacer el cálculo de valores: {e}")
                continue

            ###__________________Enviar resultados a Cacheeton____________________###
            #send_data_to_server(server_url, measurements, logger, silo)

        else:
            logger.warning("No se encontró ningún archivo .ply en el directorio.")



def main():
    #  Directorio donde se encuentran los archivos .ply de todos los silos
    data_directory = "/home/innovex/Documentos/data_tenten/silo_files/data"
    #data_directory = "/home/nicolas.donos/silo_files/data"
    output_ply_directory = "/home/innovex/Documentos/data_tenten/silo_files/view"
    silos_mesh_directory = "/home/innovex/Projects/silo_cam/labs/silos_mesh"

    # Dirección url
    sever_url = 'http://dataweb.innovex.cl:8888/'

    logger = utils.setup_logging(log_dir='logs/send_cacheton', log_file='volume_generator.log')

    # Procesar
    try:
        process_all_silos(silo_module, data_directory, output_ply_directory, silos_mesh_directory, sever_url, logger)

    except Exception as e:
        logger.error(f"Error al procesar el calculo o envío de datos: {e}")




if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)