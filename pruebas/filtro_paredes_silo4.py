"""Filtra nubes .ply eliminando SOLO las paredes del silo.

Este script se enfoca únicamente en eliminar paredes, sin aplicar
filtros adicionales que podrían eliminar contenido (pellet) del silo.

PARÁMETROS PRINCIPALES (ajustar aquí para mejores resultados):
"""
from __future__ import annotations

import argparse
import logging
import os
from typing import Optional

import numpy as np
import open3d as o3d

# ============================================================================
# PARÁMETROS DE CONFIGURACIÓN - AJUSTAR AQUÍ
# ============================================================================

# --- Método de filtrado ---
WALL_METHOD = "both"  # "cylinder", "normals", o "both"

# --- Parámetros para método CYLINDER (geométrico) ---
# Ratio del radio a conservar (0.75 = conserva 75% interior, elimina 25% exterior)
CYLINDER_RATIO = 0.75

# --- Parámetros para método NORMALS ---
# Ángulo máximo con horizontal para considerar superficie de pellet (no pared)
# Menor valor = más estricto (solo superficies muy horizontales)
# Mayor valor = más permisivo (acepta superficies más inclinadas)
HORIZONTAL_ANGLE_DEG = 35.0

# Radio de búsqueda para estimación de normales
# Menor valor = normales más locales y sensibles a detalles
# Mayor valor = normales más suaves
NORMAL_RADIUS = 0.10

# Número máximo de vecinos para estimación de normales
NORMAL_MAX_NN = 30

# --- Alineación automática ---
# Si True, detecta y rota la nube para que el eje vertical sea Z
AUTO_ALIGN = False

# Eje vertical manual (solo si AUTO_ALIGN = False)
# Opciones: "x", "y", "z"
VERTICAL_AXIS = "z"

# --- Rutas ---
DEFAULT_INPUT_DIR = "/home/gerson/Descargas/Orbbec/Orbbec_ply"
DEFAULT_OUTPUT_DIR = "/home/gerson/Descargas/Orbbec/Orbbec_filtrado"

# ============================================================================
# FIN DE PARÁMETROS - NO MODIFICAR DEBAJO DE ESTA LÍNEA
# ============================================================================




def configure_logger(verbose: bool) -> logging.Logger:
    """Configura y devuelve un logger."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, 
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger("filtro_paredes_silo")


def parse_arguments() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Elimina paredes del silo de nubes de puntos .ply"
    )
    parser.add_argument(
        "--input-dir", 
        default=DEFAULT_INPUT_DIR,
        help="Directorio con archivos .ply de entrada"
    )
    parser.add_argument(
        "--output-dir", 
        default=DEFAULT_OUTPUT_DIR,
        help="Directorio para archivos filtrados"
    )
    parser.add_argument(
        "--file", 
        dest="filename", 
        default=None,
        help="Archivo específico a procesar (nombre o ruta completa)"
    )
    parser.add_argument(
        "--list", 
        action="store_true",
        help="Lista archivos .ply disponibles y sale"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Modo verbose (más información de debug)"
    )
    parser.add_argument(
        "--visualize", 
        action="store_true", 
        help="Visualizar resultado antes de guardar"
    )
    
    return parser.parse_args()


def list_ply_files(directory: str) -> list[str]:
    """Devuelve lista de archivos .ply en el directorio."""
    try:
        entries = [
            f for f in os.listdir(directory) 
            if f.lower().endswith(".ply")
        ]
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


def load_point_cloud(path: str, logger: logging.Logger) -> o3d.geometry.PointCloud:
    """Carga una nube de puntos desde archivo."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No se encontró el archivo: {path}")
    
    logger.info(f"Cargando nube de puntos: {path}")
    pcd = o3d.io.read_point_cloud(path)
    
    if not pcd.has_points():
        raise ValueError(f"El archivo {path} no contiene puntos.")
    
    logger.info(f"Nube cargada: {len(pcd.points)} puntos")
    return pcd


def detect_vertical_axis(
    pcd: o3d.geometry.PointCloud, 
    logger: logging.Logger
) -> int:
    """
    Detecta qué eje (0=X, 1=Y, 2=Z) es el vertical.
    
    El eje vertical típicamente tiene menor dispersión ya que
    el silo es más alto que ancho.
    """
    points = np.asarray(pcd.points)
    std_devs = np.std(points, axis=0)
    
    vertical_idx = np.argmin(std_devs)
    axis_names = ['X', 'Y', 'Z']
    
    logger.info(
        f"Eje vertical detectado: {axis_names[vertical_idx]} "
        f"(dispersiones: X={std_devs[0]:.3f}, Y={std_devs[1]:.3f}, Z={std_devs[2]:.3f})"
    )
    
    return vertical_idx


def align_point_cloud(
    pcd: o3d.geometry.PointCloud, 
    vertical_axis_idx: int, 
    logger: logging.Logger
) -> o3d.geometry.PointCloud:
    """Rota la nube para que el eje vertical apunte hacia arriba (Z)."""
    if vertical_axis_idx == 2:
        logger.info("La nube ya está alineada (eje Z es vertical)")
        return pcd
    
    # Crear matriz de rotación
    if vertical_axis_idx == 0:  # X -> Z
        angle = np.pi / 2
        R = pcd.get_rotation_matrix_from_axis_angle([0, angle, 0])
        logger.info("Rotando nube: X -> Z (90° alrededor de Y)")
    else:  # Y -> Z
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
    """
    Elimina paredes conservando puntos en el interior del cilindro.
    
    Método geométrico: asume que las paredes están en el perímetro
    y el contenido en el centro.
    """
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
        f"Filtrado cilíndrico: radio máx={max_radius:.3f}m, "
        f"umbral={threshold_radius:.3f}m (ratio={ratio:.2f})"
    )
    logger.info(
        f"Puntos: {len(points)} -> {np.sum(mask)} "
        f"(eliminados {len(points) - np.sum(mask)})"
    )
    
    if np.sum(mask) == 0:
        logger.warning(
            "⚠ El filtrado cilíndrico eliminó todos los puntos. "
            "Devolviendo original."
        )
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
    """
    Elimina paredes conservando puntos con normales casi horizontales.
    
    Método de normales: las paredes tienen normales verticales (apuntan hacia afuera),
    el pellet tiene normales que apuntan principalmente hacia arriba (horizontal).
    """
    if len(pcd.points) == 0:
        return pcd

    logger.info(
        f"Estimando normales (radius={normal_radius:.3f}m, max_nn={normal_max_nn})..."
    )
    
    # Estimar normales
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius,
            max_nn=normal_max_nn,
        )
    )
    
    # NO orientar todas las normales hacia un punto (esto causa el problema)
    # En su lugar, trabajamos con las normales como están
    
    normals = np.asarray(pcd.normals)
    points = np.asarray(pcd.points)
    
    # Calcular componente vertical de las normales
    vertical_component = np.abs(normals[:, vertical_axis_idx])
    
    # NUEVO: Filtro adicional basado en la dirección radial
    # Las paredes tienen normales que apuntan hacia el centro o hacia afuera
    # El pellet tiene normales que apuntan principalmente hacia arriba
    
    # Calcular centro en el plano horizontal
    horizontal_axes = [i for i in range(3) if i != vertical_axis_idx]
    horizontal_points = points[:, horizontal_axes]
    center_2d = np.mean(horizontal_points, axis=0)
    
    # Vector desde el centro hacia cada punto (en plano horizontal)
    radial_vectors = horizontal_points - center_2d
    radial_distances = np.linalg.norm(radial_vectors, axis=1)
    
    # Normalizar vectores radiales
    valid_radial = radial_distances > 1e-6
    radial_vectors_norm = np.zeros_like(radial_vectors)
    radial_vectors_norm[valid_radial] = (
        radial_vectors[valid_radial] / radial_distances[valid_radial, np.newaxis]
    )
    
    # Componente radial de la normal (en plano horizontal)
    normals_horizontal = normals[:, horizontal_axes]
    radial_component = np.abs(
        np.sum(normals_horizontal * radial_vectors_norm, axis=1)
    )
    
    logger.info("Analizando orientación de normales...")
    logger.info(f"  Componente vertical promedio: {np.mean(vertical_component):.3f}")
    logger.info(f"  Componente radial promedio: {np.mean(radial_component):.3f}")
    
    # Criterio combinado:
    # 1. Baja componente vertical (normal apunta horizontalmente)
    # 2. Alta componente radial (normal apunta hacia/desde el centro)
    # -> Esto es una PARED
    
    # Criterio para pellet:
    # Alta componente vertical (normal apunta arriba/abajo)
    # O baja componente radial (normal no apunta hacia el centro)
    
    angle_rad = np.deg2rad(horizontal_angle_deg)
    
    # Umbral para componente vertical (ángulo con vertical)
    # Si vertical_component > cos(angle) -> es vertical -> es PELLET
    cos_threshold = np.cos(angle_rad)
    
    # Máscara para pellet: normales que apuntan principalmente arriba
    is_upward = vertical_component >= cos_threshold
    
    # También conservar puntos con normales que no apuntan radialmente
    # (incluso si son horizontales, si no apuntan al centro, son pellet)
    is_not_radial = radial_component < 0.7  # 0.7 es ~45° de desviación
    
    pellet_mask = is_upward | is_not_radial
    pellet_indices = np.where(pellet_mask)[0]
    
    logger.info(
        f"Filtrado por normales (ángulo vertical min={horizontal_angle_deg}°):"
    )
    logger.info(f"  Puntos con normales verticales: {np.sum(is_upward)}")
    logger.info(f"  Puntos con normales no-radiales: {np.sum(is_not_radial)}")
    logger.info(
        f"Puntos totales: {len(pcd.points)} | pellet conservado: {len(pellet_indices)} | "
        f"paredes eliminadas: {len(pcd.points) - len(pellet_indices)}"
    )
    
    if len(pellet_indices) == 0:
        logger.warning(
            "⚠ El filtrado por normales eliminó todos los puntos. "
            "Devolviendo original."
        )
        return pcd
    
    return pcd.select_by_index(pellet_indices)


def build_output_path(input_path: str, output_dir: Optional[str]) -> str:
    """Genera la ruta de salida para el archivo filtrado."""
    directory, filename = os.path.split(input_path)
    target_dir = output_dir or directory
    os.makedirs(target_dir, exist_ok=True)
    
    stem, ext = os.path.splitext(filename)
    return os.path.join(target_dir, f"{stem}_no_walls{ext}")


def main() -> None:
    """Función principal."""
    args = parse_arguments()
    logger = configure_logger(args.verbose)

    # Listar archivos si se solicita
    ply_files = list_ply_files(args.input_dir)
    if args.list:
        if not ply_files:
            logger.info(f"No se encontraron archivos .ply en {args.input_dir}")
        else:
            logger.info(f"Archivos disponibles en {args.input_dir}:")
            for name in ply_files:
                logger.info(f"  - {name}")
        return

    # Determinar archivo a procesar
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

    logger.info(f"📁 Archivo seleccionado: {input_path}")
    
    # Cargar nube de puntos
    point_cloud = load_point_cloud(input_path, logger)
    original_count = len(point_cloud.points)

    # Detectar y alinear eje vertical
    logger.info("=" * 60)
    logger.info("PASO 1: Detectar orientación del silo")
    logger.info("=" * 60)
    
    if VERTICAL_AXIS == "auto":
        vertical_axis_idx = detect_vertical_axis(point_cloud, logger)
    else:
        axis_map = {"x": 0, "y": 1, "z": 2}
        vertical_axis_idx = axis_map[VERTICAL_AXIS]
        logger.info(f"Eje vertical manual: {VERTICAL_AXIS.upper()}")

    if AUTO_ALIGN:
        point_cloud = align_point_cloud(point_cloud, vertical_axis_idx, logger)
        vertical_axis_idx = 2  # Después de alinear, siempre es Z

    # Aplicar filtrado de paredes
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"PASO 2: Eliminar paredes (método: {WALL_METHOD})")
    logger.info("=" * 60)
    
    filtered = point_cloud
    
    if WALL_METHOD == "cylinder":
        filtered = remove_walls_by_cylinder(
            filtered,
            ratio=CYLINDER_RATIO,
            vertical_axis_idx=vertical_axis_idx,
            logger=logger,
        )
    
    elif WALL_METHOD == "normals":
        filtered = remove_walls_by_normals(
            filtered,
            horizontal_angle_deg=HORIZONTAL_ANGLE_DEG,
            vertical_axis_idx=vertical_axis_idx,
            normal_radius=NORMAL_RADIUS,
            normal_max_nn=NORMAL_MAX_NN,
            logger=logger,
        )
    
    elif WALL_METHOD == "both":
        logger.info("Aplicando filtrado combinado...")
        logger.info("")
        logger.info("--- Paso 2a: Filtrado cilíndrico ---")
        filtered = remove_walls_by_cylinder(
            filtered,
            ratio=CYLINDER_RATIO,
            vertical_axis_idx=vertical_axis_idx,
            logger=logger,
        )
        logger.info("")
        logger.info("--- Paso 2b: Filtrado por normales ---")
        filtered = remove_walls_by_normals(
            filtered,
            horizontal_angle_deg=HORIZONTAL_ANGLE_DEG,
            vertical_axis_idx=vertical_axis_idx,
            normal_radius=NORMAL_RADIUS,
            normal_max_nn=NORMAL_MAX_NN,
            logger=logger,
        )

    # Mostrar resumen
    logger.info("")
    logger.info("=" * 60)
    logger.info("RESUMEN")
    logger.info("=" * 60)
    final_count = len(filtered.points)
    removed = original_count - final_count
    percentage_removed = (removed / original_count) * 100 if original_count > 0 else 0
    
    logger.info(f"Puntos originales:  {original_count:,}")
    logger.info(f"Puntos finales:     {final_count:,}")
    logger.info(f"Puntos eliminados:  {removed:,} ({percentage_removed:.1f}%)")

    # Visualizar si se solicita
    if args.visualize:
        logger.info("")
        logger.info("👁 Visualizando resultado...")
        o3d.visualization.draw_geometries(
            [filtered],
            window_name="Nube filtrada - Paredes eliminadas",
            width=1024,
            height=768,
        )

    # Guardar resultado
    logger.info("")
    output_path = build_output_path(input_path, args.output_dir)
    logger.info(f"💾 Guardando nube filtrada en: {output_path}")
    
    success = o3d.io.write_point_cloud(output_path, filtered, write_ascii=False)
    if not success:
        raise IOError(f"No se pudo guardar el archivo en {output_path}")

    logger.info("✅ Proceso completado correctamente.")


if __name__ == "__main__":
    main()