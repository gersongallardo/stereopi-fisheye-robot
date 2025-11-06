import logging
import os
from datetime import datetime
import numpy as np
import open3d as o3d


def setup_logging(log_dir="logs", log_function="log"):
    """
    Configura un sistema de logging
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = os.path.join(log_dir, f"{log_function}_{datetime.now().strftime('%Y%m%d')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(f'{log_function}')
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

def load_and_filter_ply(silo, filename, file_path, silos_mesh_directory):
    """
    Carga un archivo PLY y filtra los puntos según los límites y métodos especificados.
    """
    #silo_path = f"{silos_mesh_directory}/{silo.empresa}-{silo.ubicacion}_{silo.silo}_{silo.sensor}.ply"

    # Leer nube de puntos
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)

    if points is None:
        raise ValueError(f"Archivo {filename} no contiene puntos.")
    if len(points) < 3000:
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
    down_pcd = pcd.voxel_down_sample(voxel_size=0.05)

    cl, ind = down_pcd.remove_statistical_outlier(nb_neighbors=70, std_ratio=1.0)
    inlier_cloud = down_pcd.select_by_index(ind)

    return inlier_cloud