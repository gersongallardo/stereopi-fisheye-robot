import volume_calculation_of_ply_file as vc
import silos_objects as silo
import pyvista as pv
import numpy as np
import os
import pandas as pd
from datetime import datetime


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

def filtro_interpolacion_volumen(df, window_size=3, n_sigmas=3):
    df = df.copy()
    col = 'Volumen'
    k = window_size
    rolling_median = df[col].rolling(window=2*k+1, center=True).median()
    diff = np.abs(df[col] - rolling_median)
    mad = df[col].rolling(window=2*k+1, center=True).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=True
    )
    threshold = n_sigmas * mad
    outliers = diff > threshold
    df.loc[outliers, col] = np.nan
    df[col] = df[col].interpolate()
    return round(df, 2)

def crear_csv_completo(volumenes, nombres_ply, niveles, nombre_csv_file):
    df = pd.DataFrame({
        'Fecha Captura': nombres_ply,
        'Volumen': np.array(volumenes).flatten(),
        'Nivel': niveles
    })

    # Filtro + interpolación
    df = filtro_interpolacion_volumen(df)

    # Eliminar Fecha Captura
    df['Fecha Captura'] = df['Fecha Captura'].apply(lambda x: x.split("_")[1] + "_" + x.split("_")[2].replace(".ply", ""))

    # Calcular porcentaje relativo al rango de volumen
    min_vol = df['Volumen'].min()
    max_vol = df['Volumen'].max()
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
silo = silo.abtao_2_1
#silo = silo.chidhuapi3_1_1
#silo = silo.huarnorte_1_1

# Directorio
directorio = f'/home/innovex/Documentos/data_tenten/silo_files/data/{silo.empresa}/{silo.ubicacion}/{silo.silo}_{silo.sensor}/'
directorio_csv = f'/home/innovex/Documentos/data_tenten/silo_csv/{silo.empresa}/{silo.ubicacion}/'

BATCH_SIZE = 250  # Tamaño del lote

rutas_completas, nombres_completos = obtener_rutas_ordenadas(directorio)

volumenes_totales = []
nombres_totales = []
z_means_totales = []

for i in range(0, len(rutas_completas), BATCH_SIZE):
    print(f"\n=== Procesando lote {i//BATCH_SIZE + 1} ===")
    rutas_batch = rutas_completas[i:i+BATCH_SIZE]
    nombres_batch = nombres_completos[i:i+BATCH_SIZE]

    # Leer las mallas del batch
    mallas_batch = []
    for ruta in rutas_batch:
        try:
            malla = pv.read(ruta)
            mallas_batch.append(malla)
        except Exception as e:
            print(f"Error leyendo {ruta}: {e}")
            mallas_batch.append(None)

    # Filtrar fallos de lectura
    mallas_batch_filtradas = [m for m in mallas_batch if m is not None]
    nombres_batch_filtrados = [nombre for m, nombre in zip(mallas_batch, nombres_batch) if m is not None]

    # Procesar mallas y calcular volumen
    mallas_proc, nombres_validos, z_means = arreglo_de_mallas_procesadas(mallas_batch_filtradas, nombres_batch_filtrados, silo)
    volumenes = get_results_of_volume_meassurement(mallas_proc, z_means, silo)

    # Acumular resultados
    volumenes_totales.extend(volumenes)
    nombres_totales.extend(nombres_validos)
    z_means = [round(z + silo.alto_total,2) for z in z_means]
    z_means_totales.extend(z_means)

# Crear CSV final
crear_csv_completo(volumenes_totales, nombres_totales, z_means_totales, f'{directorio_csv}/{silo.empresa}-{silo.ubicacion}_{silo.silo}_{silo.sensor}.csv')
