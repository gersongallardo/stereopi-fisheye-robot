import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import silos_objects as silo
import numpy as np

def graficar_volumen(df, silo):
    # Asegurar que la columna es datetime
    df['Fecha Captura'] = pd.to_datetime(df['Fecha Captura'])

    # Crear el gráfico de líneas con los círculos
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['Fecha Captura'],
        y=df['Volumen'],
        mode='markers+lines',
        marker=dict(
            size=8,
            color='rgb(100, 255, 50)',
            symbol='circle',
            line=dict(width=2, color='rgb(0, 0, 0)')
        ),
        line=dict(width=4, color='rgb(255, 0, 0)')
    ))

    fig.update_layout(
        title=f'Volumen de pellet en el silo {silo.silo} {silo.ubicacion}.',
        title_x=0.5,
        title_font=dict(family='Arial', size=24, color='rgb(37, 37, 37)'),

        xaxis_title='Fecha de captura (día)',
        yaxis_title='Volumen (m\u00b3)',
        xaxis_title_font=dict(family='Arial', size=20, color='rgb(37, 37, 37)'),
        yaxis_title_font=dict(family='Arial', size=20, color='rgb(37, 37, 37)'),

        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Arial', size=12, color='rgb(37, 37, 37)'),
        margin=dict(l=50, r=50, t=50, b=50),
        hovermode='x unified',
        showlegend=False,
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),

        xaxis=dict(
            type='date',
            showgrid=True,
            gridcolor='black',
            gridwidth=0.5,
            tickformat='%Y-%m-%d',
            tickangle=45,
            dtick=86400000,  # 1 día en milisegundos
            hoverformat='%Y-%m-%d %H:%M:%S'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='black',
            gridwidth=0.5,
            dtick=10
        ),
    )

    fig.show()


def graficar_toneladas(df):
    # Crear el gráfico de líneas con los círculos
    fig = go.Figure()

    # Añadir la línea con los círculos marcando cada punto
    fig.add_trace(go.Scatter(
        x=df['Fecha Captura'],
        y=df['Tons'],
        mode='markers+lines',  # 'markers+lines' para mostrar tanto las líneas como los puntos
        marker=dict(
            size=8,  # Tamaño de los círculos
            color='rgb(100, 255, 50)',  # Color de los círculos
            symbol='circle',  # Forma del marcador (círculo)
            line=dict(width=2, color='rgb(0, 0, 0)')  # Borde de los círculos
        ),
        line=dict(width=4, color='rgb(255, 0, 0)')  # Estilo de la línea
    ))

    # Actualizar el diseño para hacerlo más profesional y dinámico
    fig.update_layout(
        title='Toneladas en Silo 2 Abtao',  # Título del gráfico
        title_x=0.5,  # Centrar el título
        title_font=dict(family='Arial', size=18, color='rgb(37, 37, 37)'),  # Fuente del título
        # Ejes
        xaxis_title='Fecha de captura del dato',  # Nombre eje X
        yaxis_title='Toneladas (Tons) ',  # Nombre eje Y
        xaxis_title_font=dict(family='Arial', size=20, color='rgb(37, 37, 37)'),  # Fuente eje X
        yaxis_title_font=dict(family='Arial', size=20, color='rgb(37, 37, 37)'),  # Fuente eje Y
        # Estilo de fondo
        plot_bgcolor='white',  # Fondo del gráfico
        paper_bgcolor='white',  # Fondo de la página
        font=dict(family='Arial', size=12, color='rgb(37, 37, 37)'),  # Fuente de los textos
        margin=dict(l=50, r=50, t=50, b=50),  # Márgenes del gráfico
        # Interactividad
        hovermode='x unified',  # Mostrar todos los valores a la vez cuando se pasa el ratón por encima
        showlegend=False,  # Desactivar la leyenda si solo hay una serie de datos
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),  # Estilo de la etiqueta al hacer hover
        # Configuración de la grilla
        xaxis=dict(
            showgrid=True,  # Mostrar grilla en el eje X
            gridcolor='black',  # Color de la grilla
            gridwidth=0.5,  # Grosor de las líneas de la grilla
            dtick=5  # Espaciado entre las líneas de la grilla (ajustar según sea necesario)
        ),
        yaxis=dict(
            showgrid=True,  # Mostrar grilla en el eje Y
            gridcolor='black',  # Color de la grilla
            gridwidth=0.5,  # Grosor de las líneas de la grilla
            dtick=10  # Espaciado entre las líneas de la grilla (ajustar según sea necesario)
        ),
    )

    # Mostrar el gráfico
    fig.show()

def graficar_porcentaje(df, silo):
    def calcular_porcentaje(row):
        # Obtener valor máximo del volumen
        max_volume = df['Volumen'].max()
        min_volume = df['Volumen'].min()
        range_volume = max_volume - min_volume
        return round(((row['Volumen'] - min_volume) / range_volume) * 100,1)

    df['Porcentaje'] = df.apply(calcular_porcentaje, axis=1)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['Fecha Captura'],
        y=df['Porcentaje'],
        mode='markers+lines',
        marker=dict(
            size=8,
            color='rgb(100, 255, 50)',
            symbol='circle',
            line=dict(width=2, color='rgb(0, 0, 0)')
        ),
        line=dict(width=4, color='rgb(0, 0, 255)')
    ))

    fig.update_layout(
        title=f'Porcentaje de pellet en el silo {silo.silo} {silo.ubicacion}',
        title_x=0.5,
        title_font=dict(family='Arial', size=24, color='rgb(37, 37, 37)'),
        xaxis_title='Fecha de captura (día)',
        yaxis_title='Porcentaje (%)',
        xaxis_title_font=dict(family='Arial', size=20, color='rgb(37, 37, 37)'),
        yaxis_title_font=dict(family='Arial', size=20, color='rgb(37, 37, 37)'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Arial', size=12, color='rgb(37, 37, 37)'),
        margin=dict(l=50, r=50, t=50, b=50),
        hovermode='x unified',
        showlegend=False,
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
        xaxis=dict(
            type='date',
            showgrid=True,
            gridcolor='black',
            gridwidth=0.5,
            dtick=86400000,  # un día en milisegundos
            tickformat='%Y-%m-%d',
            hoverformat='%Y-%m-%d %H:%M:%S',
            tickangle=45
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='black',
            gridwidth=0.5,
            dtick=10
        ),
    )
    fig.show()


def graficar_diferencia_metodos(df1, df2):
    fig = go.Figure()

    # Primer método (rojo)
    fig.add_trace(go.Scatter(
        x=df1['Fecha Captura'],
        y=df1['Volumen'],
        mode='markers+lines',
        name='Volumen',
        marker=dict(
            size=8,
            color='rgb(255, 0, 0)',
            symbol='circle',
            line=dict(width=2, color='rgb(0, 0, 0)')
        ),
        line=dict(width=4, color='rgb(255, 0, 0)')
    ))

    # Segundo método (azul)
    fig.add_trace(go.Scatter(
        x=df2['Fecha Captura'],
        y=df2['Nivel'],
        mode='markers+lines',
        name='Nivel',
        marker=dict(
            size=8,
            color='rgb(0, 0, 255)',
            symbol='circle',
            line=dict(width=2, color='rgb(0, 0, 0)')
        ),
        line=dict(width=4, color='rgb(0, 0, 255)')
    ))

    # Layout actualizado
    fig.update_layout(
        title='Comparación de Volumen en silo Chidhuapi3 por métodos',
        title_x=0.5,
        title_font=dict(family='Arial', size=24, color='rgb(37, 37, 37)'),
        xaxis_title='Fecha de captura (día)',
        yaxis_title='Volumen (m\u00b3)',
        xaxis_title_font=dict(family='Arial', size=20, color='rgb(37, 37, 37)'),
        yaxis_title_font=dict(family='Arial', size=20, color='rgb(37, 37, 37)'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Arial', size=12, color='rgb(37, 37, 37)'),
        margin=dict(l=50, r=50, t=50, b=50),
        hovermode='x unified',
        showlegend=True,
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
        xaxis=dict(
            type='date',
            showgrid=True,
            gridcolor='black',
            gridwidth=0.5,
            dtick=86400000,  # cada 1 día
            tickformat='%Y-%m-%d',
            hoverformat='%Y-%m-%d %H:%M:%S',
            tickangle=45
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='black',
            gridwidth=0.5,
            dtick=10
        ),
    )

    fig.show()


def procesar_fechas(df):
    # Extraer y convertir a datetime
    df['Fecha Captura'] = pd.to_datetime(
        df['Fecha Captura'].str.extract(r'(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})')[0],
        format='%Y-%m-%d_%H:%M:%S'
    )
    return df

def filtro_interpolacion_volumen(df, window_size=3, n_sigmas=3):
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

def filtrar_una_muestra_cada_n_horas(df, n=0):
    # Filtrar solo donde los minutos sean 0
    df = df[df['Fecha Captura'].dt.minute == 0].copy()

    # Crear columna de agrupación
    df['Hora Agrupada'] = df['Fecha Captura'].dt.floor(f'{n}h')

    # Ordenar y agrupar
    df = df.sort_values('Fecha Captura')
    df = df.groupby('Hora Agrupada').first().reset_index()

    # Eliminar o renombrar la columna original duplicada si existe
    if 'Fecha Captura' in df.columns and 'Hora Agrupada' in df.columns:
        df = df.drop(columns=['Fecha Captura'])

    # Renombrar para graficar
    df = df.rename(columns={'Hora Agrupada': 'Fecha Captura'})

    return df


# Seleccionar silo
#silo = silo.abtao_2_1
#silo = silo.chidhuapi3_1_1
silo = silo.huarnorte_1_1

# Path de los CSV
#csv_path_media = f'/home/innovex/Documentos/data_rpi3/CSVs/conteo_puntos_plys_completo.csv'
#csv_path_media = f'/home/innovex/Documentos/data_rpi3/CSVs/Abtao_media_completa.csv'
#csv_path_media = f'/home/innovex/Documentos/data_rpi3/test/node_1/Chidhuapi3_media_ajuste_sigmoidal.csv'
#csv_path_media = f'/home/innovex/Documentos/data_rpi3/CSVs/Chidhuapi3_conteo_puntos_plys_completo.csv'
csv_path_media = f'/home/innovex/Documentos/data_tenten/silo_csv/{silo.empresa}/{silo.ubicacion}/{silo.empresa}-{silo.ubicacion}_{silo.silo}_{silo.sensor}.csv'

df_media = pd.read_csv(csv_path_media)
#df_1_punto = pd.read_csv(csv_path_1_punto)

# Procesar fechas en ambos DataFrames
df_media = procesar_fechas(df_media)
#df_1_punto = procesar_fechas(df_1_punto)

# Filtrar muestras por hora
#df_media = filtrar_una_muestra_cada_n_horas(df_media)
#df_1_punto = filtrar_una_muestra_cada_n_horas(df_1_punto)

# Interpolar y limpiar datos
df_media = filtro_interpolacion_volumen(df_media, window_size=20, n_sigmas=2)

# Graficar volumen
#graficar_volumen(df_media, silo)
graficar_volumen(df_media, silo)

graficar_diferencia_metodos(df_media,df_media)
#graficar_toneladas(df_media)

# Graficar porcentaje
#graficar_porcentaje(df_media, silo)
graficar_porcentaje(df_media, silo)