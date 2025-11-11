"""Utility script to experiment with clustering-based point cloud filters.

This module mirrors the automated behaviour of :mod:`ply_filter` but is aimed at
manual experimentation. It loads a PLY file from the tester directory and applies
filters inspired by ``oak8_clustering.py`` allowing the parameters to be tweaked
through command line arguments.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import open3d as o3d


DEFAULT_INPUT_DIR = "/home/gerson/Documentos/plys_tester"
DEFAULT_OUTPUT_DIR = "/home/gerson/Documentos/filter/"


@dataclass
class FilterConfig:
    """Container holding filtering parameters."""

    enable_clustering: bool = True
    cluster_eps: float = 0.5  #elimina a los lados (mover entre 0.4 y 0.6 o no reduce ruido)
    cluster_min_points: int = 150
    downsample_voxel_size: float = 0.05
    enable_statistical_filter: bool = True
    stat_nb_neighbors: int = 80
    stat_std_ratio: float = 3.5 # reduce en z(+) pero, si se reduce mucho deja espacios huevos


def configure_logger(verbose: bool) -> logging.Logger:
    """Create and return a configured logger."""

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger("test_filter_ply")


def parse_arguments() -> tuple[argparse.Namespace, FilterConfig]:
    """Parse CLI arguments and produce a :class:`FilterConfig`."""

    parser = argparse.ArgumentParser(
        description=(
            "Filtra nubes de puntos existentes utilizando los mismos parámetros "
            "que oak8_clustering.py, permitiendo ajustarlos desde la línea de comandos."
        )
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help="Directorio que contiene los archivos .ply a evaluar.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directorio en el que se guardará la nube filtrada. Por defecto se "
            "utiliza /home/gerson/Documentos/filter/."
        ),
    )
    parser.add_argument(
        "--file",
        dest="filename",
        default=None,
        help=(
            "Nombre del archivo .ply a procesar. Si no se indica se utilizará el "
            "más reciente encontrado en el directorio de entrada."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Muestra los archivos .ply disponibles y termina.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Activa registros detallados durante el procesamiento.",
    )

    default_config = FilterConfig()

    # Parámetros de filtrado equivalentes a oak8_clustering.py
    parser.add_argument(
        "--cluster-eps",
        type=float,
        default=default_config.cluster_eps,
        help="Distancia máxima entre puntos de un mismo cluster (DBSCAN).",
    )
    parser.add_argument(
        "--cluster-min-points",
        type=int,
        default=default_config.cluster_min_points,
        help="Número mínimo de puntos para considerar un cluster válido.",
    )
    parser.add_argument(
        "--no-clustering",
        action="store_true",
        help="Deshabilita el filtrado por clustering.",
    )
    parser.add_argument(
        "--downsample-voxel-size",
        type=float,
        default=default_config.downsample_voxel_size,
        help="Tamaño del voxel para la reducción de densidad (downsampling).",
    )
    parser.add_argument(
        "--no-downsample",
        action="store_true",
        help="Evita el paso de downsampling previo al filtrado.",
    )
    parser.add_argument(
        "--stat-nb-neighbors",
        type=int,
        default=default_config.stat_nb_neighbors,
        help="Número de vecinos para el filtro estadístico de outliers.",
    )
    parser.add_argument(
        "--stat-std-ratio",
        type=float,
        default=default_config.stat_std_ratio,
        help="Ratio de desviación estándar para el filtro estadístico.",
    )
    parser.add_argument(
        "--no-statistical-filter",
        action="store_true",
        help="Deshabilita el filtrado estadístico de outliers.",
    )

    args = parser.parse_args()

    config = FilterConfig(
        enable_clustering=not args.no_clustering,
        cluster_eps=args.cluster_eps,
        cluster_min_points=args.cluster_min_points,
        downsample_voxel_size=args.downsample_voxel_size,
        enable_statistical_filter=not args.no_statistical_filter,
        stat_nb_neighbors=args.stat_nb_neighbors,
        stat_std_ratio=args.stat_std_ratio,
    )

    if args.no_downsample:
        config.downsample_voxel_size = 0.0

    return args, config


def list_ply_files(directory: str) -> list[str]:
    """Return a sorted list with the available PLY files in *directory*."""

    try:
        entries = [f for f in os.listdir(directory) if f.lower().endswith(".ply")]
    except FileNotFoundError:
        return []

    entries.sort()
    return entries


def pick_latest_ply(directory: str) -> Optional[str]:
    """Return the path to the most recently modified PLY file in *directory*."""

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
    """Load a point cloud from a PLY file raising informative errors."""

    if not os.path.isfile(path):
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    point_cloud = o3d.io.read_point_cloud(path)
    if not point_cloud.has_points():
        raise ValueError(f"El archivo {path} no contiene puntos.")
    return point_cloud


def apply_filters(pcd: o3d.geometry.PointCloud, config: FilterConfig, logger: logging.Logger) -> o3d.geometry.PointCloud:
    """Apply downsampling, statistical filtering and clustering to *pcd*."""

    logger.info("Puntos iniciales: %s", len(pcd.points))

    # Downsampling
    if config.downsample_voxel_size > 0:
        logger.info("Aplicando downsampling (voxel_size=%.3f)...", config.downsample_voxel_size)
        pcd = pcd.voxel_down_sample(voxel_size=config.downsample_voxel_size)
        logger.info("Puntos tras downsampling: %s", len(pcd.points))

    # Statistical outlier removal
    if config.enable_statistical_filter and len(pcd.points) > 0:
        logger.info(
            "Aplicando filtro estadístico (nb_neighbors=%s, std_ratio=%.2f)...",
            config.stat_nb_neighbors,
            config.stat_std_ratio,
        )
        _, indices = pcd.remove_statistical_outlier(
            nb_neighbors=config.stat_nb_neighbors,
            std_ratio=config.stat_std_ratio,
        )
        pcd = pcd.select_by_index(indices)
        logger.info("Puntos tras filtro estadístico: %s", len(pcd.points))

    # Clustering (DBSCAN)
    if config.enable_clustering and len(pcd.points) > 0:
        logger.info(
            "Aplicando clustering DBSCAN (eps=%.3f, min_points=%s)...",
            config.cluster_eps,
            config.cluster_min_points,
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
            logger.warning("No se encontraron clusters válidos; se conserva la nube original tras los pasos previos.")
            return pcd

        unique_labels, counts = np.unique(labels[valid_mask], return_counts=True)
        largest_label = unique_labels[np.argmax(counts)]
        logger.info(
            "Cluster seleccionado: %s (puntos=%s) de un total de %s clusters",
            largest_label,
            counts.max(),
            len(unique_labels),
        )
        indices = np.where(labels == largest_label)[0]
        pcd = pcd.select_by_index(indices)
        logger.info("Puntos tras clustering: %s", len(pcd.points))

    return pcd


def build_output_path(input_path: str, output_dir: Optional[str]) -> str:
    """Return the destination path for the filtered point cloud."""

    directory, filename = os.path.split(input_path)
    target_dir = output_dir or directory
    os.makedirs(target_dir, exist_ok=True)

    stem, ext = os.path.splitext(filename)
    return os.path.join(target_dir, f"{stem}_filtered{ext}")


def main() -> None:
    args, config = parse_arguments()
    logger = configure_logger(args.verbose)

    ply_files = list_ply_files(args.input_dir)
    if args.list:
        if not ply_files:
            logger.info("No se encontraron archivos .ply en %s", args.input_dir)
        else:
            logger.info("Archivos disponibles en %s:", args.input_dir)
            for name in ply_files:
                logger.info(" - %s", name)
        return

    if args.filename:
        input_path = args.filename
        if not os.path.isabs(input_path):
            input_path = os.path.join(args.input_dir, input_path)
    else:
        latest = pick_latest_ply(args.input_dir)
        if latest is None:
            raise FileNotFoundError(
                f"No se encontraron archivos .ply en {args.input_dir}. Usa --list para verificar."
            )
        input_path = latest

    logger.info("Archivo seleccionado: %s", input_path)
    point_cloud = load_point_cloud(input_path)

    filtered = apply_filters(point_cloud, config, logger)

    output_path = build_output_path(input_path, args.output_dir)
    logger.info("Guardando nube filtrada en: %s", output_path)
    success = o3d.io.write_point_cloud(output_path, filtered, write_ascii=False)
    if not success:
        raise IOError(f"No se pudo guardar el archivo filtrado en {output_path}")

    logger.info("Proceso completado correctamente.")


if __name__ == "__main__":
    main()
