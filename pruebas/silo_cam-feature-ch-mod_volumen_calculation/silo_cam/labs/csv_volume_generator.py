import volume_calculation_of_ply_file as vc
import silos_objects as silo
import pyvista as pv
import numpy as np
import os
import pandas as pd
from datetime import datetime
import open3d as o3d


def obtener_rutas_ordenadas(carpeta="node_0"):
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
            return datetime.strptime(f"{fecha_str} {hora_str}", "%Y-%m-%d %H:%M:%S")
        except Exception as e:
            print(f"No se pudo extraer fecha y hora de {nombre_archivo}: {e}")
            return datetime.min

    archivos.sort(key=extraer_fecha_hora)
    nombres = [os.path.basename(f) for f in archivos]
    return archivos, nombres

def filtro_interpolacion_volumen(df, window_size=15, n_sigmas=3):
    # Filtro de valores atípicos e interpolación
    df = df.copy()
    col = 'Volumen'
    # Aplicar filtro de Hampel
    series = df[col]
    k = window_size
    rolling_median = series.rolling(window=2*k+1, center=True).median()
    diff = np.abs(series - rolling_median)
    mad = series.rolling(window=2*k+1, center=True).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=True
    )
    threshold = n_sigmas * mad
    outliers = diff > threshold

    # Reemplazar outliers por NaN e interpolar
    df.loc[outliers, col] = np.nan
    df[col] = df[col].interpolate()

    # Redondear el resultado
    df[col] = df[col].round(0)

    return df

def crear_csv_completo(volumenes, nombres_ply, niveles, nombre_csv_file):
    df = pd.DataFrame({
        'Fecha Captura': nombres_ply,
        'Volumen': np.array(volumenes).flatten(),
        'Nivel': niveles
    })

    # Filtro + interpolación
    #df = filtro_interpolacion_volumen(df)

    # Eliminar Fecha Captura
    df['Fecha Captura'] = df['Fecha Captura'].apply(lambda x: x.split("_")[1] + "_" + x.split("_")[2].replace(".ply", ""))

    # Calcular porcentaje relativo al rango de volumen
    #min_vol = df['Volumen'].min()
    min_vol = 0
    #max_vol = df['Volumen'].max()
    max_vol = 78.36
    df['Porcentaje'] = df['Volumen'].apply(lambda v: round(((v - min_vol) / (max_vol - min_vol)) * 100, 1))

    # Calcular toneladas
    densidad_pellet = 0.734  # ton/m3
    df['Toneladas'] = df['Volumen'].apply(lambda v: round(v * densidad_pellet, 1))

    # Calcular nivel
    #df['Nivel'] = df['Nivel'].apply(lambda n: round(n, 2))

    # Reordenar columnas
    df = df[['Fecha Captura', 'Volumen', 'Porcentaje', 'Toneladas', 'Nivel']]

    # Guardar
    df.to_csv(nombre_csv_file, index=False)
    print(f"CSV creado: {nombre_csv_file}")


def load_and_filter_ply(silo, file_path, silos_mesh_directory, logger=None, visualizar=False):
    """
    Carga, filtra y limpia una nube de puntos PLY según los parámetros del silo.
    Retorna la nube filtrada (Open3D PointCloud).
    """
    filename = os.path.basename(file_path)
    silo_path = f"{silos_mesh_directory}/{silo.empresa}-{silo.ubicacion}_{silo.silo}_{silo.sensor}.ply"

    # Leer nube de puntos
    try:
        pcd = o3d.io.read_point_cloud(file_path)
    except Exception as e:
        raise ValueError(f"Error leyendo {filename}: {e}")

    points = np.asarray(pcd.points)
    if points is None or len(points) == 0:
        raise ValueError(f"Archivo {filename} no contiene puntos.")
    #if len(points) < 30000:
    #    raise ValueError(f"Archivo {filename} con pocos puntos ({len(points)}).")

    # Aplicar traslación y rotación
    points += np.array(silo.traslacion)
    R = pcd.get_rotation_matrix_from_xyz(np.deg2rad(silo.rotacion))
    pcd.rotate(R, center=(0, 0, 0))

    # Filtrado por límites
    try:
        mask = (
            (points[:, 0] >= silo.x_min) & (points[:, 0] <= silo.x_max) &
            (points[:, 1] >= silo.y_min) & (points[:, 1] <= silo.y_max) &
            (points[:, 2] >= silo.z_min) & (points[:, 2] <= silo.z_max)
        )
        filtered_points = points[mask]
    except Exception as e:
        raise ValueError(f"Error filtrando puntos de {filename}: {e}")

    # Crear nueva nube filtrada
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_points)

    # Downsampling y eliminación de ruido
    down_pcd = pcd.voxel_down_sample(voxel_size=0.05)
    cl, ind = down_pcd.remove_statistical_outlier(nb_neighbors=70, std_ratio=1.0)
    inlier_cloud = down_pcd.select_by_index(ind)

    # Detección de clusters (mantener el mayor)
    labels = np.array(inlier_cloud.cluster_dbscan(eps=0.2, min_points=30, print_progress=False))
    if labels.max() < 0:
        if logger:
            logger.warning(f"No se encontraron clusters en {filename}. Se devuelve nube filtrada original.")
        return inlier_cloud

    counts = np.bincount(labels[labels >= 0])
    largest_cluster_id = np.argmax(counts)
    inlier_cloud = inlier_cloud.select_by_index(np.where(labels == largest_cluster_id)[0])

    # Opcional: Visualización
    if visualizar:
        try:
            silo_mesh = o3d.io.read_triangle_mesh(silo_path)
            silo_mesh.translate([0, 0, -silo.alto_total])
            pcd_silo_mesh = silo_mesh.sample_points_uniformly(number_of_points=200000)
            pcd_silo_mesh.paint_uniform_color([0.6, 0.6, 0.6])
            o3d.visualization.draw_geometries([inlier_cloud, pcd_silo_mesh],
                                              window_name=f"Filtered {filename}")
        except Exception as e:
            if logger:
                logger.warning(f"No se pudo visualizar {filename}: {e}")

    return inlier_cloud


def arreglo_de_mallas_procesadas(arreglo_de_mallas_reducido, nombres_ply_reducido, silo_object):
    mallas_procesadas = []
    nombres_validos = []
    z_means = []
    for i in range(len(arreglo_de_mallas_reducido)):
        try:
            print(f"Procesando malla [{i+1}/{len(arreglo_de_mallas_reducido)}]")
            malla_procesada, z_mean = vc.proccesing_mesh_with_z_mean(arreglo_de_mallas_reducido[i], silo_object)
            mallas_procesadas.append(malla_procesada)
            nombres_validos.append(nombres_ply_reducido[i])  # solo si fue exitosa
            z_means.append(z_mean)
        except Exception as e:
            print(f"Error al procesar la malla [{i+1}: {e}]")
    return mallas_procesadas, nombres_validos, z_means


def get_results_of_volume_meassurement(mallas_procesadas, z_mean, silo_object):
    # Inicializar un arreglo vacío para almacenar los resultados
    resultados = []

    # Recorrer cada malla procesada y obtener los valores
    for i in range(len(mallas_procesadas)):
        try:
            print(f"Estimando volumen malla [{i+1}/{len(mallas_procesadas)}]")
            # Calculo de volumen según forma de silo
            if silo_object.alto_prisma == None:
                volumen_silo = vc.volume_measurement_of_silo(mallas_procesadas[i], z_mean[i], silo_object)
            else:
                volumen_silo = vc.volume_measurement_of_silo_with_prism(mallas_procesadas[i], z_mean[i], silo_object)

            # Guardar los valores en el arreglo
            resultados.append([volumen_silo])
        except Exception as e:
            print(f"Error al estimar volumen malla [{i+1}: {e}]")
            resultados.append([0])

    # Convertir el arreglo de resultados en una lista de 3 columnas (mallas_procesadas x 3)
    resultados = np.array(resultados)
    return resultados

# Seleccionar el silo a utilizar
silo = silo.abtao_4_1
#silo = silo.chidhuapi3_1_1
#silo = silo.huarnorte_1_1

# Directorio
#directorio = f'/home/innovex/Documentos/data_tenten/silo_files/data/{silo.empresa}/{silo.ubicacion}/{silo.silo}_{silo.sensor}/'
#directorio_csv = f'/home/innovex/Documentos/data_tenten/silo_csv/{silo.empresa}/{silo.ubicacion}/'

directorio = f'/home/innovex/Escritorio/calibracion//{silo.silo}_{silo.sensor}/'
directorio_csv = f'/home/innovex/Escritorio/calibracion/'
silos_mesh_directory = "/home/innovex/Projects/silo_cam/labs/silos_mesh"

BATCH_SIZE = 250  # Tamaño del lote

rutas_completas, nombres_completos = obtener_rutas_ordenadas(directorio)

volumenes_totales = []
nombres_totales = []
z_means_totales = []

for i in range(0, len(rutas_completas), BATCH_SIZE):
    print(f"\n=== Procesando lote {i // BATCH_SIZE + 1} ===")
    rutas_batch = rutas_completas[i:i + BATCH_SIZE]
    nombres_batch = nombres_completos[i:i + BATCH_SIZE]

    mallas_sin_ruido = []
    nombres_validos_batch = []

    for ruta, nombre in zip(rutas_batch, nombres_batch):
        try:
            nube_filtrada = load_and_filter_ply(
                silo, ruta, silos_mesh_directory, logger=None, visualizar=True
            )

            # Convertir Open3D → PyVista
            puntos = np.asarray(nube_filtrada.points)
            if len(puntos) == 0:
                print(f"⚠️ Nube vacía en {nombre}, se omite.")
                continue

            malla_pv = pv.PolyData(puntos)
            mallas_sin_ruido.append(malla_pv)
            nombres_validos_batch.append(nombre)

        except Exception as e:
            print(f"❌ Error procesando {nombre}: {e}")

    # Si no hay nubes válidas en el batch, continuar
    if not mallas_sin_ruido:
        print("⚠️ No se procesaron nubes válidas en este lote.")
        continue

    # Procesar mallas y calcular volumen
    mallas_proc, nombres_validos, z_means = arreglo_de_mallas_procesadas(
        mallas_sin_ruido, nombres_validos_batch, silo
    )

    volumenes = get_results_of_volume_meassurement(mallas_proc, z_means, silo)

    # Acumular resultados
    volumenes_totales.extend(volumenes)
    nombres_totales.extend(nombres_validos)
    z_means_ajustados = [round(z + silo.alto_total, 2) for z in z_means]
    z_means_totales.extend(z_means_ajustados)

# Crear CSV final
crear_csv_completo(
    volumenes_totales,
    nombres_totales,
    z_means_totales,
    f'{directorio_csv}/{silo.empresa}-{silo.ubicacion}_{silo.silo}_{silo.sensor}.csv'
)
