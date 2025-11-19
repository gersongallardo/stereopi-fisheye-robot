"""Filtra nubes .ply para eliminar paredes del silo usando múltiples estrategias.

Estrategias implementadas:
1. Filtrado geométrico por cilindro (elimina puntos cerca del perímetro)
2. Filtrado por normales (conserva superficies horizontales, elimina verticales)
3. Auto-detección de orientación de la nube
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import open3d as o3d

DEFAULT_INPUT_DIR = "/home/gerson/Descargas/Orbbec/Orbbec_ply"
DEFAULT_OUTPUT_DIR = "/home/gerson/Descargas/Orbbec/Orbbec_filtrado"


@dataclass
class FilterConfig:
    """Parámetros básicos de filtrado."""

    enable_clustering: bool = True
    cluster_eps: float = 0.5
    cluster_min_points: int = 150
    downsample_voxel_size: float = 0.05
    enable_statistical_filter: bool = True
    stat_nb_neighbors: int = 80
    stat_std_ratio: float = 3.5


@dataclass
class WallRemovalConfig:
    """Parámetros para eliminar paredes del silo."""

    method: str = "cylinder"  # "cylinder", "normals", "both"
    
    # Para método geométrico (cilindro)
    cylinder_ratio: float = 0.85  # Conservar puntos en el 85% interior del radio
    
    # Para método de normales
    vertical_axis: str = "auto"  # "auto", "x", "y", "z"
    horizontal_angle_deg: float = 30.0  # Ángulo máx para considerar superficie horizontal
    normal_radius: float = 0.08
    normal_max_nn: int = 30
    
    # Rotación automática
    auto_align: bool = True


def configure_logger(verbose: bool) -> logging.Logger:
    """Devuelve un logger configurado."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger("filtro_paredes_silo")


def parse_arguments() -> tuple[argparse.Namespace, FilterConfig, WallRemovalConfig]:
    """Parsea argumentos y devuelve configuraciones."""
    parser = argparse.ArgumentParser(
        description="Filtra nubes de puntos eliminando paredes del silo."
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--file", dest="filename", default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--visualize", action="store_true", help="Visualizar resultado antes de guardar")

    default_config = FilterConfig()
    parser.add_argument("--cluster-eps", type=float, default=default_config.cluster_eps)
    parser.add_argument("--cluster-min-points", type=int, default=default_config.cluster_min_points)
    parser.add_argument("--no-clustering", action="store_true")
    parser.add_argument("--downsample-voxel-size", type=float, default=default_config.downsample_voxel_size)
    parser.add_argument("--no-downsample", action="store_true")
    parser.add_argument("--stat-nb-neighbors", type=int, default=default_config.stat_nb_neighbors)
    parser.add_argument("--stat-std-ratio", type=float, default=default_config.stat_std_ratio)
    parser.add_argument("--no-statistical-filter", action="store_true")

    # Eliminación de paredes
    parser.add_argument(
        "--wall-method",
        choices=["cylinder", "normals", "both"],
        default="cylinder",
        help="Método para eliminar paredes: cylinder (geométrico), normals, o both"
    )
    parser.add_argument(
        "--cylinder-ratio",
        type=float,
        default=0.75,
        help="Ratio del radio para conservar (0.85 = conservar 85%% interior)"
    )
    parser.add_argument(
        "--vertical-axis",
        choices=["auto", "x", "y", "z"],
        default="auto",
        help="Eje vertical (auto = detección automática)"
    )
    parser.add_argument(
        "--horizontal-angle-deg",
        type=float,
        default=30.0,
        help="Ángulo máximo con horizontal para conservar punto (pellet)"
    )
    parser.add_argument("--wall-normal-radius", type=float, default=0.08)
    parser.add_argument("--wall-normal-max-nn", type=int, default=30)
    parser.add_argument("--no-auto-align", action="store_true", help="No alinear automáticamente la nube")

    args = parser.parse_args()

    filter_config = FilterConfig(
        enable_clustering=not args.no_clustering,
        cluster_eps=args.cluster_eps,
        cluster_min_points=args.cluster_min_points,
        downsample_voxel_size=args.downsample_voxel_size,
        enable_statistical_filter=not args.no_statistical_filter,
        stat_nb_neighbors=args.stat_nb_neighbors,
        stat_std_ratio=args.stat_std_ratio,
    )

    if args.no_downsample:
        filter_config.downsample_voxel_size = 0.0

    wall_config = WallRemovalConfig(
        method=args.wall_method,
        cylinder_ratio=args.cylinder_ratio,
        vertical_axis=args.vertical_axis,
        horizontal_angle_deg=args.horizontal_angle_deg,
        normal_radius=args.wall_normal_radius,
        normal_max_nn=args.wall_normal_max_nn,
        auto_align=not args.no_auto_align,
    )

    return args, filter_config, wall_config


def list_ply_files(directory: str) -> list[str]:
    """Devuelve los .ply disponibles."""
    try:
        entries = [f for f in os.listdir(directory) if f.lower().endswith(".ply")]
    except FileNotFoundError:
        return []
    entries.sort()
    return entries


def pick_latest_ply(directory: str) -> Optional[str]:
    """Devuelve la ruta del archivo .ply más reciente."""
    try:
        candidates = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.lower().endswith(".ply")
        ]
    except FileNotFoundError:
        return None
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def load_point_cloud(path: str) -> o3d.geometry.PointCloud:
    """Carga una nube de puntos."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No se encontró el archivo: {path}")
    pcd = o3d.io.read_point_cloud(path)
    if not pcd.has_points():
        raise ValueError(f"El archivo {path} no contiene puntos.")
    return pcd


def detect_vertical_axis(pcd: o3d.geometry.PointCloud, logger: logging.Logger) -> int:
    """Detecta qué eje (0=X, 1=Y, 2=Z) es el vertical basándose en la dispersión."""
    points = np.asarray(pcd.points)
    std_devs = np.std(points, axis=0)
    
    # El eje vertical suele tener menor dispersión (el silo es más alto que ancho)
    vertical_idx = np.argmin(std_devs)
    axis_names = ['X', 'Y', 'Z']
    logger.info(f"Eje vertical detectado: {axis_names[vertical_idx]} (dispersiones: X={std_devs[0]:.3f}, Y={std_devs[1]:.3f}, Z={std_devs[2]:.3f})")
    
    return vertical_idx


def align_point_cloud(pcd: o3d.geometry.PointCloud, vertical_axis_idx: int, logger: logging.Logger) -> o3d.geometry.PointCloud:
    """Rota la nube para que el eje vertical detectado apunte hacia arriba (Z)."""
    if vertical_axis_idx == 2:
        logger.info("La nube ya está alineada correctamente (eje Z es vertical)")
        return pcd
    
    # Crear matriz de rotación para alinear el eje vertical con Z
    if vertical_axis_idx == 0:  # X -> Z
        # Rotar 90° alrededor de Y
        angle = np.pi / 2
        R = pcd.get_rotation_matrix_from_axis_angle([0, angle, 0])
        logger.info("Rotando nube: X -> Z (90° alrededor de Y)")
    else:  # Y -> Z
        # Rotar -90° alrededor de X
        angle = -np.pi / 2
        R = pcd.get_rotation_matrix_from_axis_angle([angle, 0, 0])
        logger.info("Rotando nube: Y -> Z (-90° alrededor de X)")
    
    pcd_aligned = pcd.rotate(R, center=(0, 0, 0))
    return pcd_aligned


def remove_walls_by_cylinder(
    pcd: o3d.geometry.PointCloud,
    ratio: float,
    vertical_axis_idx: int,
    logger: logging.Logger,
) -> o3d.geometry.PointCloud:
    """Elimina paredes conservando puntos en el interior del cilindro."""
    if len(pcd.points) == 0:
        return pcd
    
    points = np.asarray(pcd.points)
    
    # Identificar los dos ejes horizontales
    horizontal_axes = [i for i in range(3) if i != vertical_axis_idx]
    
    # Calcular centro y radio en el plano horizontal
    horizontal_points = points[:, horizontal_axes]
    center_2d = np.mean(horizontal_points, axis=0)
    
    # Distancias desde el centro en el plano horizontal
    distances = np.linalg.norm(horizontal_points - center_2d, axis=1)
    max_radius = np.percentile(distances, 95)  # Usar percentil para evitar outliers
    
    # Conservar puntos dentro del ratio especificado del radio
    threshold_radius = max_radius * ratio
    mask = distances <= threshold_radius
    
    logger.info(
        f"Filtrado cilíndrico: radio máx={max_radius:.3f}m, umbral={threshold_radius:.3f}m"
    )
    logger.info(
        f"Puntos: {len(points)} -> {np.sum(mask)} (eliminados {len(points) - np.sum(mask)})"
    )
    
    if np.sum(mask) == 0:
        logger.warning("El filtrado cilíndrico eliminó todos los puntos. Devolviendo original.")
        return pcd
    
    return pcd.select_by_index(np.where(mask)[0])


def remove_walls_by_normals(
    pcd: o3d.geometry.PointCloud,
    horizontal_angle_deg: float,
    vertical_axis_idx: int,
    normal_radius: float,
    normal_max_nn: int,
    logger: logging.Logger,
) -> o3d.geometry.PointCloud:
    """Elimina paredes conservando puntos con normales casi horizontales (pellet)."""
    if len(pcd.points) == 0:
        return pcd

    logger.info(
        f"Estimando normales (radius={normal_radius:.3f}m, max_nn={normal_max_nn})..."
    )
    
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius,
            max_nn=normal_max_nn,
        )
    )
    
    # Orientar normales consistentemente hacia el viewpoint (arriba)
    viewpoint = np.array([0.0, 0.0, 1000.0])
    if vertical_axis_idx == 0:
        viewpoint = np.array([1000.0, 0.0, 0.0])
    elif vertical_axis_idx == 1:
        viewpoint = np.array([0.0, 1000.0, 0.0])
    
    pcd.orient_normals_towards_camera_location(viewpoint)
    
    normals = np.asarray(pcd.normals)
    
    # Componente vertical de las normales
    vertical_component = np.abs(normals[:, vertical_axis_idx])
    
    # Ángulo con respecto a la horizontal
    # Si vertical_component es pequeño -> normal casi horizontal -> pellet
    # Si vertical_component es grande -> normal casi vertical -> pared
    angle_rad = np.deg2rad(horizontal_angle_deg)
    sin_threshold = np.sin(angle_rad)
    
    # Conservar puntos cuya normal forme un ángulo pequeño con la horizontal
    pellet_mask = vertical_component <= sin_threshold
    pellet_indices = np.where(pellet_mask)[0]
    
    logger.info(
        f"Filtrado por normales (ángulo max={horizontal_angle_deg}°):"
    )
    logger.info(
        f"Puntos totales: {len(pcd.points)} | pellet: {len(pellet_indices)} | "
        f"paredes eliminadas: {len(pcd.points) - len(pellet_indices)}"
    )
    
    if len(pellet_indices) == 0:
        logger.warning("El filtrado por normales eliminó todos los puntos. Devolviendo original.")
        return pcd
    
    return pcd.select_by_index(pellet_indices)


def apply_filters(
    pcd: o3d.geometry.PointCloud,
    config: FilterConfig,
    logger: logging.Logger
) -> o3d.geometry.PointCloud:
    """Aplica downsampling, filtro estadístico y clustering."""
    logger.info(f"Puntos iniciales: {len(pcd.points)}")

    if config.downsample_voxel_size > 0:
        logger.info(f"Aplicando downsampling (voxel_size={config.downsample_voxel_size:.3f})...")
        pcd = pcd.voxel_down_sample(voxel_size=config.downsample_voxel_size)
        logger.info(f"Puntos tras downsampling: {len(pcd.points)}")

    if config.enable_statistical_filter and len(pcd.points) > 0:
        logger.info(
            f"Aplicando filtro estadístico (nb_neighbors={config.stat_nb_neighbors}, "
            f"std_ratio={config.stat_std_ratio:.2f})..."
        )
        _, indices = pcd.remove_statistical_outlier(
            nb_neighbors=config.stat_nb_neighbors,
            std_ratio=config.stat_std_ratio,
        )
        pcd = pcd.select_by_index(indices)
        logger.info(f"Puntos tras filtro estadístico: {len(pcd.points)}")

    if config.enable_clustering and len(pcd.points) > 0:
        logger.info(
            f"Aplicando clustering DBSCAN (eps={config.cluster_eps:.3f}, "
            f"min_points={config.cluster_min_points})..."
        )
        labels = np.array(
            pcd.cluster_dbscan(
                eps=config.cluster_eps,
                min_points=config.cluster_min_points,
                print_progress=False,
            )
        )

        valid_mask = labels >= 0
        if not np.any(valid_mask):
            logger.warning("No se encontraron clusters válidos.")
            return pcd

        unique_labels, counts = np.unique(labels[valid_mask], return_counts=True)
        largest_label = unique_labels[np.argmax(counts)]
        logger.info(
            f"Cluster seleccionado: {largest_label} (puntos={counts.max()}) "
            f"de {len(unique_labels)} clusters"
        )
        indices = np.where(labels == largest_label)[0]
        pcd = pcd.select_by_index(indices)
        logger.info(f"Puntos tras clustering: {len(pcd.points)}")

    return pcd


def build_output_path(input_path: str, output_dir: Optional[str]) -> str:
    """Genera la ruta de salida."""
    directory, filename = os.path.split(input_path)
    target_dir = output_dir or directory
    os.makedirs(target_dir, exist_ok=True)
    stem, ext = os.path.splitext(filename)
    return os.path.join(target_dir, f"{stem}_filtered{ext}")


def main() -> None:
    args, filter_config, wall_config = parse_arguments()
    logger = configure_logger(args.verbose)

    ply_files = list_ply_files(args.input_dir)
    if args.list:
        if not ply_files:
            logger.info(f"No se encontraron archivos .ply en {args.input_dir}")
        else:
            logger.info(f"Archivos disponibles en {args.input_dir}:")
            for name in ply_files:
                logger.info(f" - {name}")
        return

    if args.filename:
        input_path = args.filename
        if not os.path.isabs(input_path):
            input_path = os.path.join(args.input_dir, input_path)
    else:
        latest = pick_latest_ply(args.input_dir)
        if latest is None:
            raise FileNotFoundError(
                f"No se encontraron archivos .ply en {args.input_dir}"
            )
        input_path = latest

    logger.info(f"Archivo seleccionado: {input_path}")
    point_cloud = load_point_cloud(input_path)

    # Aplicar filtros básicos
    filtered = apply_filters(point_cloud, filter_config, logger)

    # Detectar y alinear eje vertical
    if wall_config.vertical_axis == "auto":
        vertical_axis_idx = detect_vertical_axis(filtered, logger)
    else:
        axis_map = {"x": 0, "y": 1, "z": 2}
        vertical_axis_idx = axis_map[wall_config.vertical_axis]
        logger.info(f"Eje vertical especificado manualmente: {wall_config.vertical_axis.upper()}")

    if wall_config.auto_align:
        filtered = align_point_cloud(filtered, vertical_axis_idx, logger)
        # Después de alinear, el eje vertical es siempre Z
        vertical_axis_idx = 2

    # Aplicar filtrado de paredes
    logger.info(f"=== Eliminando paredes usando método: {wall_config.method} ===")
    
    if wall_config.method == "cylinder":
        filtered = remove_walls_by_cylinder(
            filtered,
            ratio=wall_config.cylinder_ratio,
            vertical_axis_idx=vertical_axis_idx,
            logger=logger,
        )
    elif wall_config.method == "normals":
        filtered = remove_walls_by_normals(
            filtered,
            horizontal_angle_deg=wall_config.horizontal_angle_deg,
            vertical_axis_idx=vertical_axis_idx,
            normal_radius=wall_config.normal_radius,
            normal_max_nn=wall_config.normal_max_nn,
            logger=logger,
        )
    elif wall_config.method == "both":
        # Primero filtrado cilíndrico (más agresivo)
        filtered = remove_walls_by_cylinder(
            filtered,
            ratio=wall_config.cylinder_ratio,
            vertical_axis_idx=vertical_axis_idx,
            logger=logger,
        )
        # Luego filtrado por normales (refinamiento)
        filtered = remove_walls_by_normals(
            filtered,
            horizontal_angle_deg=wall_config.horizontal_angle_deg,
            vertical_axis_idx=vertical_axis_idx,
            normal_radius=wall_config.normal_radius,
            normal_max_nn=wall_config.normal_max_nn,
            logger=logger,
        )

    # Visualizar si se solicita
    if args.visualize:
        logger.info("Visualizando resultado...")
        o3d.visualization.draw_geometries(
            [filtered],
            window_name="Nube filtrada",
            width=1024,
            height=768,
        )

    # Guardar resultado
    output_path = build_output_path(input_path, args.output_dir)
    logger.info(f"Guardando nube filtrada en: {output_path}")
    success = o3d.io.write_point_cloud(output_path, filtered, write_ascii=False)
    if not success:
        raise IOError(f"No se pudo guardar el archivo en {output_path}")

    logger.info("✓ Proceso completado correctamente.")


if __name__ == "__main__":
    main()