import logging
import os
from datetime import datetime
import numpy as np
import open3d as o3d

import volume_calculation_of_ply_file as vc
import calendar
import requests


def setup_logging(log_dir="logs", log_file="log"):
    """
    Configura un sistema de logging
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = os.path.join(log_dir, f"{log_file}_{datetime.now().strftime('%Y%m%d')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(f'{log_file}')
    logger.setLevel(logging.INFO)
    return logger

def extract_timestamp(filename):
    """
    Extraer timestamp de archivo PLY.
    """
    parts = filename.split('_')
    time_str = parts[2].replace('.ply', '')
    date_str = f"{parts[1]} {time_str}"
    return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")

def last_file_ply(silo, ply_directory):
    """
    Extraer el último archivo PLY de un silo dado.
    """
    ply_path = f"{ply_directory}/{silo.empresa}/{silo.ubicacion}/{silo.silo}_{silo.sensor}"

    if not os.path.isdir(ply_path):
        raise FileNotFoundError(f"Directorio no encontrado: {ply_path}")

    ply_files = [f for f in os.listdir(ply_path) if f.endswith(".ply")]
    if not ply_files:
        raise FileNotFoundError(f"No se encontraron archivos .ply en {ply_path}")

    # Ordenar por timestamp y tomar el más reciente
    try:
        ply_files.sort(key=extract_timestamp)
        filename = ply_files[-1]
        return filename, os.path.join(ply_path, filename)
    except (ValueError, TypeError, IndexError) as e:
        raise ValueError(f"Error al ordenar archivos por timestamp: {e}")


###_________________Calculo de volumen_______________________###

def ply_point_filter(points, x_lim=(-3, 3), y_lim=(-3, 3), z_lim=(-6, -0.32)):
    """
    Filtra puntos de la superficie dentro de los límites especificados.
    """
    if points is None:
        raise ValueError("El arreglo de puntos es None.")
    if points.size == 0:
        raise ValueError("El arreglo de puntos está vacío.")
    if not isinstance(points, np.ndarray):
        raise TypeError(f"Se esperaba un numpy.ndarray, se recibió: {type(points)}")

    mask = (
        (points[:, 0] >= x_lim[0]) & (points[:, 0] <= x_lim[1]) &
        (points[:, 1] >= y_lim[0]) & (points[:, 1] <= y_lim[1]) &
        (points[:, 2] >= z_lim[0]) & (points[:, 2] <= z_lim[1])
    )
    return points[mask]

def load_and_filter_ply(silo, filename, file_path, silos_mesh_directory, logger):
    """
    Carga un archivo PLY y filtra los puntos según los límites y métodos especificados.
    """
    silo_path = f"{silos_mesh_directory}/{silo.empresa}-{silo.ubicacion}_{silo.silo}_{silo.sensor}.ply"

    # Leer nube de puntos
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)

    if points is None:
        raise ValueError(f"Archivo {filename} no contiene puntos.")
    if len(points) < 30000:
        raise ValueError(f"Archivo {filename} con pocos puntos ({len(points)}).")

    # Aplicar centrado de la nube en 0,0
    points += np.array(silo.traslacion)

    # Aplicar la rotación
    R = pcd.get_rotation_matrix_from_xyz(np.deg2rad(silo.rotacion))
    pcd.rotate(R, center=(0, 0, 0))

    # Convertir a numpy para seguir con el procesamiento
    rotated_points = np.asarray(pcd.points)

    try:
        # Filtrar puntos
        filtered_points = ply_point_filter(rotated_points,
            x_lim=(silo.x_min, silo.x_max),
            y_lim=(silo.y_min, silo.y_max),
            z_lim=(silo.z_min, silo.z_max)
        )
    except ValueError as e:
        raise ValueError(f"Error en filtrado de puntos por límites: {e}")

    # Crar nube filtrada
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_points)

    # Downsampling y filtrado estadístico
    #down_pcd = pcd.uniform_down_sample(every_k_points=60)
    down_pcd = pcd.voxel_down_sample(voxel_size=0.05)

    cl, ind = down_pcd.remove_statistical_outlier(nb_neighbors=70, std_ratio=1.0)
    #cl, ind = down_pcd.remove_radius_outlier(nb_points=50, radius=0.15)
    inlier_cloud = down_pcd.select_by_index(ind)
    #cl, ind = inlier_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2)
    #inlier_cloud = inlier_cloud.select_by_index(ind)

    labels = np.array(inlier_cloud.cluster_dbscan(eps=0.2, min_points=30, print_progress=False))
    if labels.max() < 0:
        logger.warning(f"No se encontraron clusters en {filename}. Se devuelve nube filtrada original.")
        return inlier_cloud

    counts = np.bincount(labels[labels >= 0])
    largest_cluster_id = np.argmax(counts)
    inlier_cloud = inlier_cloud.select_by_index(np.where(labels == largest_cluster_id)[0])

    # Lectura de silo
    silo_mesh = o3d.io.read_triangle_mesh(silo_path)

    # Aplicar traslación en z al silo
    translation_silo_mesh = np.array([0, 0, -silo.alto_total])
    silo_mesh.translate(translation_silo_mesh)

    # Convertir silo mesh a pcd
    pcd_silo_mesh = silo_mesh.sample_points_uniformly(number_of_points=1000000)
    # Por ejemplo, color gris claro (valores entre 0 y 1)
    single_color = np.array([[0.6, 0.6, 0.6]])  # RGB
    # Repite el color para cada punto
    pcd_silo_mesh.colors = o3d.utility.Vector3dVector(np.tile(single_color, (len(pcd_silo_mesh.points), 1)))

    # Visualización
    o3d.visualization.draw_geometries([pcd, pcd_silo_mesh], zoom=0.8, window_name=f"PCD {filename}")
    o3d.visualization.draw_geometries([down_pcd], zoom=0.6, front=[0, -1, 1], window_name=f"Down PCD {filename}")
    o3d.visualization.draw_geometries([inlier_cloud], zoom=0.6, front=[0, -1, 1], window_name=f"Filter PCD {filename}")

    return inlier_cloud


def filter_and_save_ply(filename, file_path, output_ply_directory, silos_mesh_directory, logger, silo):
    """
    Procesa un archivo PLY: carga, filtra y guarda el resultado.
    """
    filtered_ply = None
    silo_name = f"{silo.empresa}-{silo.ubicacion}_{silo.silo}_{silo.sensor}"

    try:
        # Filtrar el archivo PLY
        filtered_ply = load_and_filter_ply(silo, filename, file_path, silos_mesh_directory, logger)
        logger.info(f"Archivo filtrado.")

        # Guardar el archivo filtrado
        output_path = os.path.join(output_ply_directory,  f'{silo_name}.ply')

        # Verificar que el directorio de salida existe
        if not os.path.isdir(output_ply_directory):
            raise FileNotFoundError(f"Directorio de salida no encontrado: {output_ply_directory}")

        success = o3d.io.write_point_cloud(output_path, filtered_ply, compressed=True)
        if not success:
            raise IOError(f"No se pudo guardar el archivo PLY en {output_path}.")
        logger.info(f"Archivo guardado en: {output_path}")

        return filtered_ply

    except FileNotFoundError as fnf_error:
        logger.error(f"Archivo no encontrado: {fnf_error}")
    except IOError as ioe:
        logger.error(f"Error de escritura al guardar archivo: {ioe}")
    except ValueError as ve:
        logger.error(f"Valor incorrecto al procesar {silo_name}: {ve}")
    except Exception as e:
        logger.error(f"Error inesperado al procesar {silo_name}: {e}")

    return filtered_ply

###_________________Calculo de volumen_______________________###

def mesh_processing(mesh, filename, silo_object):
    """
    Procesa una malla para extruirla y calcula el valor de altura medio (z).
    """
    try:
        processed_mesh, z_mean = vc.proccesing_mesh_with_z_mean(mesh, silo_object)
        return processed_mesh, z_mean
    except Exception as e:
        raise ValueError(f"Error al procesar la malla '{filename}': {e}")

def get_results_of_volume_meassurement(processed_mesh, z_mean, silo_object):
    """
    Calcula el volumen de la malla procesada.
    """
    try:
        # Calculo de volumen según forma de silo
        if silo_object.alto_prisma == None:
            pellet_volume = vc.volume_measurement_of_silo(processed_mesh, z_mean, silo_object)
        else:
            pellet_volume = vc.volume_measurement_of_silo_with_prism(processed_mesh, z_mean, silo_object)
    except Exception as e:
        raise ValueError(f"Error al estimar volumen de la malla: {e}")
    return pellet_volume

def calculate_values(filename, volume, level, silo_object):
    """
    Calcula los valores de porcentaje y toneladas del pellet con base al volumen y nivel de la malla procesada.
    """
    # Convertir nombres a fechas UNIX
    date_str = filename.split("_")[1] + "_" + filename.split("_")[2].replace(".ply", "")
    date_unix = int(calendar.timegm(datetime.strptime(date_str, "%Y-%m-%d_%H:%M:%S").timetuple()))  # GMT +0000
    #date_unix = int(time.mktime(datetime.strptime(date_str, "%Y-%m-%d_%H:%M:%S").timetuple()))     # GMT -0400

    # Calcular porcentaje relativo
    min_vol = silo_object.volumen_minimo
    max_vol = silo_object.volumen_maximo
    percentage = round(((volume - min_vol) / (max_vol- min_vol)) * 100, 1)

    # Calcular toneladas
    pellet_density = 0.734
    tons = round(volume * pellet_density, 1)

    # Diccionario de valores
    status = 0
    measurements = {'unixtime': date_unix,
                 'volume': volume,
                 'porcentage': percentage,
                 'tons': tons,
                 'level': level,
                 'status': status}

    return measurements

def send_data_to_server(server_url, measurements, logger, silo_object):
    """
    Envía los datos al servidor mediante una solicitud PUT.
    """
    try:
        url = f"{server_url}/{silo_object.empresa}-{silo_object.ubicacion}/slevel/{silo_object.silo}/{silo_object.sensor}/"

        # Parámetros
        params = {
            'time': measurements['unixtime'],
            'volume': measurements['volume'],
            'porcentage': measurements['porcentage'],
            'level': measurements['level'],
            'fill': measurements['tons'],
            'status': measurements['status']
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