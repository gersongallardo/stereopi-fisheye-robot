import numpy as np
import pyvista as pv
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from scipy.interpolate import Rbf
from pykrige.rk import Krige
from scipy.spatial import KDTree
from scipy.spatial import cKDTree
import silos_objects as silo

DENSIDAD_PELLET            = 0.734
VOLUMEN_TOTAL_SILO_ABTAO   = 78.36

INTERVALO_ARREGLO = 10

################ INICIALIZACIÓN DE PARAMETROS DE PROCESAMIENTO ####################
DEGREE_OF_POLYNOM_ITERATION_1 = 2 #estaba en 2
DEGREE_OF_POLYNOM_ITERATION_2 = 3

#LIMITE_EXTRAPOLACION_ITEARACION_1 = 2
#LIMITE_EXTRAPOLACION_RF = 4 #estaba en 4
#LIMITE_EXTRAPOLACION_ITEARACION_2 =5



#####################################################################################
# /**
#  * @brief Load a .ply file and assign a gradient based on Z elevation.
#  *
#  * This function reads a .ply file using PyVista, and adds an "Elevation" scalar
#  * to the mesh based on the Z-coordinate of each point. This can be useful for
#  * visualizing the geometry using color gradients.
#  *
#  * @param file_path Path to the .ply file to load.
#  * @return pyvista.PolyData object with an "Elevation" scalar array added.
#  */uracy and performance.
def load_and_plot_ply_with_gradient(file_path):
    mesh = pv.read(file_path)
    mesh["Elevation"] = mesh.points[:, 2]  # Use Z coordinate for gradient
    return mesh


# /**
#  * @brief Plot a mesh with labeled axes.
#  *
#  * This function uses PyVista's Plotter to display a mesh with a color gradient
#  * based on the "Elevation" scalar. It also adds labeled axes and a title.
#  *
#  * @param mesh A pyvista.PolyData object containing the mesh to be plotted.
#  *             It must include an "Elevation" scalar for color mapping.
#  * @return None
#  */
def plotter_mesh(mesh):
    plotter = pv.Plotter()

    # Add mesh with color gradient
    plotter.add_mesh(mesh, scalars="Elevation", cmap="viridis")

    # Add labeled axes (usando argumentos correctos)
    plotter.show_bounds(
        grid="front",         # Mostrar una grilla de fondo
        location="outer",     # Etiquetas en los bordes exteriores
        xtitle="X (meters)",  # en lugar de xlabel
        ytitle="Y (meters)",  # en lugar de ylabel
        ztitle="Z (meters)",  # en lugar de zlabel
        font_size=10          # argumento válido para tamaño de fuente
    )

    # Agregar texto al gráfico
    plotter.add_text("Point Cloud Mesh", font_size=10)  # font_size correcto

    plotter.show()


# /**
#  * @brief Plot the extrapolated and original mesh with volume and capture date.
#  *
#  * This function displays two meshes using PyVista: the extrapolated mesh with a color
#  * gradient based on elevation and the original mesh in red. It also shows the estimated
#  * volume as on-screen text and labels the axes.
#  *
#  * @param mesh_extrapolada The extrapolated mesh (pyvista.PolyData) with an "Elevation" scalar.
#  * @param mesh_original The original mesh (pyvista.PolyData), shown in red.
#  * @param volumen Estimated volume in cubic meters (float).
#  * @param fecha_captura Capture date or label to display as text (string).
#  * @return None
#  */
def plotter_mesh_extrapolated_and_original(mesh_extrapoled, mesh_original, volume, capture_date):
    plotter = pv.Plotter()

    # Show estimated volume as on-screen text
    text_volume = f"Estimated Volume: {volume:.0f} m³"
    plotter.add_text(text_volume, font_size_of_surface=12, position='upper_right')

    # Configure scalar bar arguments
    scalar_bar_args = {
        "title": "Elevation",
        "vertical": True,
        "title_font_size_of_surface": 12,
        "label_font_size_of_surface": 10,
        "position_x": 0.85,  # Horizontal offset (0 to 1)
        "position_y": 0.25,  # Vertical position
        "width": 0.05,
        "height": 0.5
    }

    # Add the extrapolated mesh with elevation-based color gradient
    plotter.add_mesh(mesh_extrapoled, scalars="Elevation", cmap="viridis", scalar_bar_args=scalar_bar_args)

    # Add the original mesh in red
    plotter.add_mesh(mesh_original, color="red")

    # Add labeled axes
    plotter.show_bounds(
        grid="front",
        location="outer",
        xlabel="X (meters)",
        ylabel="Y (meters)",
        zlabel="Z (meters)",
        font_size_of_surface=10,
    )
    plotter.add_text(capture_date, font_size_of_surface=10)
    plotter.show()

# /**
#  * @brief Plot a mesh with labeled axes and volume-related measurements.
#  *
#  * This function displays a mesh with an elevation-based color gradient and
#  * overlays textual information such as the estimated volume, estimated tonnage,
#  * and the percentage fill relative to the total silo volume.
#  *
#  * @param mesh A pyvista.PolyData object representing the 3D mesh with an "Elevation" scalar.
#  * @param volumen Raw volume before subtracting the inverted pyramid (float, in m³).
#  * @return None
#  */
def plotter_mesh_with_volume_meassurement(mesh, volume, silo_object):
    volume = volume - silo_object.vol_compl_piram
    tons = volume * DENSIDAD_PELLET
    percentage_volume = (volume / VOLUMEN_TOTAL_SILO_ABTAO) * 100

    plotter = pv.Plotter()

    # Display volume, tonnage and fill percentage as on-screen text
    text_volume = f"Estimated Volume: {volume:.0f} m³"
    text_tonnage = f"Estimated Tonnage: {tons:.0f} T"
    text_percentage = f"Silo Fill: {percentage_volume:.0f}%"

    plotter.add_text(text_tonnage, font_size_of_surface=12, position='upper_right')
    plotter.add_text(text_volume, font_size_of_surface=12, position='upper_left')
    plotter.add_text(text_percentage, font_size_of_surface=12, position='lower_left')

    # Add mesh with elevation-based color gradient
    plotter.add_mesh(mesh, scalars="Elevation", cmap="viridis")

    # Add labeled axes
    plotter.show_bounds(
        grid="front",      # Show a background grid
        location="outer",  # Show labels on the outer edges
        xlabel="X (meters)",
        ylabel="Y (meters)",
        zlabel="Z (meters)",
        font_size_of_surface=10,
    )
    plotter.show()

# /**
#  * @brief Create a horizontal plane at a given Z position.
#  *
#  * This function creates a plane in the XY plane, centered at the origin,
#  * with a configurable position along the Z axis. The plane has dimensions
#  * 5x5 units and a resolution of 1x1.
#  *
#  * @param posicion_eje_z Position along the Z-axis where the plane will be placed (float).
#  * @return A pyvista.PolyData object representing the plane.
#  */
def create_plane_ply(position_axis_z,silo_object):
    # Create a plane in the XY plane centered at origin, with size_of_surface 5x5
    plane = pv.Plane(
        center=(0, 0, position_axis_z),  # Z-axis position
        direction=(0, 0, 1),            # Normal vector (perpendicular to Z)
        i_size =silo_object.largo,                       # size_of_surface in X
        j_size =silo_object.ancho,                       # size_of_surface in Y
        i_resolution=1,
        j_resolution=1
    )
    return plane

# /**
#  * @brief Project a mesh onto a plane located below its center.
#  *
#  * This function computes a plane located one-third of the mesh's height
#  * below its center along the Z-axis, and projects the mesh onto that plane.
#  *
#  * @param mesh A pyvista.PolyData object representing the input 3D mesh.
#  * @return A pyvista.PolyData object containing the projected mesh.
#  */
def projection_to_a_plane(mesh):
    origin = mesh.center
    origin[-1] -= mesh.length / 3.0  # Shift Z downward by one-third of total mesh height
    projection = mesh.project_points_to_plane(origin=origin)
    return projection

# /**
#  * @brief Plot and compare the original mesh with two modified versions.
#  *
#  * This function creates a 3-panel visualization showing the original mesh
#  * and two modified versions, each with their corresponding segmented regions
#  * (e.g., full, medium, and empty) overlaid in red. It also displays estimated
#  * volumes and average distances for each configuration.
#  *
#  * @param mesh_original The original 3D mesh (pyvista.PolyData).
#  * @param mesh_modificada First modified mesh.
#  * @param mesh_modificada_2 Second modified mesh.
#  * @param name_of_mesh_original Label for the original mesh.
#  * @param type_of_modification Label for the first modification.
#  * @param type_of_modification_2 Label for the second modification.
#  * @param volumen_lleno Estimated volume for the full configuration.
#  * @param volumen_medio Estimated volume for the medium configuration.
#  * @param volumen_vacio Estimated volume for the empty configuration.
#  * @param mesh_lleno Red overlay mesh for full configuration.
#  * @param mesh_medio Red overlay mesh for medium configuration.
#  * @param mesh_vacio Red overlay mesh for empty configuration.
#  * @param distancia_lleno Average distance for full configuration.
#  * @param distancia_medio Average distance for medium configuration.
#  * @param distancia_vacio Average distance for empty configuration.
#  * @return None
#  */
def ploteo_comparacion_de_mesh(
    mesh_original,
    mesh_modified,
    mesh_modified_2,
    name_of_mesh_original,
    type_of_modification,
    type_of_modification_2,
    volume_full,
    volume_half,
    volume_empty,
    mesh_full,
    mesh_half,
    mesh_empty,
    distance_full,
    distance_half,
    distance_empty
):
    # Volume text
    text_vol_full = f"Estimated Volume: {volume_full:.0f} m³"
    text_vol_med = f"Estimated Volume: {volume_half:.0f} m³"
    text_vol_empty = f"Estimated Volume: {volume_empty:.0f} m³"

    # Distance text
    text_dist_full = f"Average Distance: {distance_full:.2f} m"
    text_dist_med = f"Average Distance: {distance_half:.2f} m"
    text_dist_empty = f"Average Distance: {distance_empty:.2f} m"

    plotter = pv.Plotter(shape=(1, 3))  # 1 row, 3 columns

    # Subplot 1: Original mesh
    plotter.subplot(0, 0)
    plotter.add_mesh(mesh_original, scalars="Elevation", cmap="viridis")
    plotter.add_mesh(mesh_full, color="red")
    plotter.add_text(name_of_mesh_original, font_size_of_surface=10)
    plotter.add_text(text_vol_full, font_size_of_surface=12, position='upper_right')
    plotter.add_text(text_dist_full, font_size_of_surface=12, position='lower_right')
    plotter.show_bounds(
        grid="front", location="outer",
        xtitle="X (meters)", ytitle="Y (meters)", ztitle="Z (meters)", font_size_of_surface=10,
    )

    # Subplot 2: First modified mesh
    plotter.subplot(0, 1)
    plotter.add_mesh(mesh_modified, scalars="Elevation", cmap="viridis")
    plotter.add_mesh(mesh_half, color="red")
    plotter.add_text(type_of_modification, font_size_of_surface=10)
    plotter.add_text(text_vol_med, font_size_of_surface=12, position='upper_right')
    plotter.add_text(text_dist_med, font_size_of_surface=12, position='lower_right')
    plotter.show_bounds(
        grid="front", location="outer",
        xtitle="X (meters)", ytitle="Y (meters)", ztitle="Z (meters)", font_size_of_surface=10,
    )

    # Subplot 3: Second modified mesh
    plotter.subplot(0, 2)
    plotter.add_mesh(mesh_modified_2, scalars="Elevation", cmap="viridis")
    plotter.add_mesh(mesh_empty, color="red")
    plotter.add_text(type_of_modification_2, font_size_of_surface=10)
    plotter.add_text(text_vol_empty, font_size_of_surface=12, position='upper_right')
    plotter.add_text(text_dist_empty, font_size_of_surface=12, position='lower_right')
    plotter.show_bounds(
        grid="front", location="outer",
        xtitle="X (meters)", ytitle="Y (meters)", ztitle="Z (meters)", font_size_of_surface=10,
    )
    plotter.show()

# /**
#  * @brief Create a structured surface mesh from 3D coordinate arrays.
#  *
#  * Generates a PyVista StructuredGrid mesh from the provided X, Y, and Z coordinate arrays.
#  * The mesh includes an "Elevation" scalar based on the Z-values, useful for color mapping.
#  *
#  * @param xx A 2D NumPy array of X coordinates.
#  * @param yy A 2D NumPy array of Y coordinates.
#  * @param zz A 2D NumPy array of Z coordinates.
#  * @return pv.StructuredGrid A PyVista structured surface mesh with elevation scalars.
#  */
def create_surface_mesh(xx, yy, zz):
    surface = pv.StructuredGrid(xx, yy, zz)
    surface["Elevation"] = surface.points[:, 2]
    return surface


# /**
#  * @brief Extrapolates a surface using polynomial regression over a square grid.
#  *
#  * This function fits a polynomial surface to a given mesh, then extrapolates it
#  * over a square grid of specified size_of_surface. The grid's resolution and polynomial degree
#  * are configurable. The resulting surface is returned as a structured grid.
#  *
#  * @param mesh The original PyVista mesh to be extrapolated.
#  * @param degree_of_polynom Degree of the polynomial regression model.
#  * @param size_of_surface The size_of_surface of the grid over which the surface is extrapolated.
#  * @param resolution The resolution (number of points) for the grid.
#  * @return pv.StructuredGrid A structured surface mesh with the extrapolated surface.
#  */
def extrapolate_polynomial_surface(mesh, degree_of_polynom, silo_object, resolution):
    points = mesh.points
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    X = np.column_stack((x, y))

    # Fit a polynomial regression model
    poly = PolynomialFeatures(degree=degree_of_polynom)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, z)

    # Create a square grid with specified resolution
    x_lin = np.linspace(-silo_object.largo / 2, silo_object.largo / 2, resolution)
    y_lin = np.linspace(-silo_object.ancho / 2, silo_object.ancho / 2, resolution)
    xx, yy = np.meshgrid(x_lin, y_lin)
    xy_mesh = np.column_stack((xx.ravel(), yy.ravel()))

    # Predict Z values for the grid
    Z_pred = model.predict(poly.transform(xy_mesh))
    zz = Z_pred.reshape(resolution, resolution)

    return create_surface_mesh(xx, yy, zz)

# /**
#  * @brief Extrapolates a surface using a Random Forest Regressor.
#  *
#  * This function fits a Random Forest model to the given mesh points, then extrapolates
#  * the surface over a square grid of specified size_of_surface. The resulting surface is returned as
#  * a structured grid. The Random Forest model is configured with hyperparameters to prevent
#  * overfitting and improve generalization.
#  *
#  * @param mesh The original PyVista mesh to be extrapolated.
#  * @param size_of_surface The size_of_surface of the square grid over which the surface is extrapolated.
#  * @param resolution The resolution (number of points) for the grid.
#  * @return pv.StructuredGrid A structured surface mesh with the extrapolated surface.
#  */
def extrapolate_surface_rf(mesh, silo_object, resolution):
    assert hasattr(silo_object, 'largo'), "silo_object debe ser una instancia de la clase Silo"

    points = mesh.points
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    X = np.column_stack((x, y))

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=4,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, z)

    x_lin = np.linspace(-silo_object.largo / 2, silo_object.largo / 2, resolution)
    y_lin = np.linspace(-silo_object.ancho / 2, silo_object.ancho / 2, resolution)
    xx, yy = np.meshgrid(x_lin, y_lin)
    xy_mesh = np.column_stack((xx.ravel(), yy.ravel()))

    Z_pred = model.predict(xy_mesh)
    zz = Z_pred.reshape(resolution, resolution)

    return create_surface_mesh(xx, yy, zz)


# /**
#  * @brief Extrudes a mesh surface along a defined direction.
#  *
#  * This function extrudes the given mesh surface in the positive Z direction
#  * by creating a plane at a specified offset and trimming the mesh to match
#  * the defined plane. The extruded mesh is then returned.
#  *
#  * @param mesh The PyVista mesh to be extruded.
#  * @return pv.PolyData The extruded mesh after applying the plane trimming.
#  */
def extrude_ply_surface(mesh,silo_object):
    # Create a plane at a specified position to define the extrusion bounds
    plane = pv.Plane(
    center=(mesh.center[0], mesh.center[1], - silo_object.alto_total),
    direction=(0, 0, -1),
    i_size=30,
    j_size=30,
    )

    # Perform the extrusion of the mesh along the Z direction
    mesh_extruded = mesh.extrude_trim((0, 0, 1), plane)
    mesh_extruded["Elevation"] = mesh_extruded.points[:, 2]  # Use Z coordinate for gradient
    return mesh_extruded


# /**
#  * @brief Filters the mesh points where Z is less than or equal to 0.
#  *
#  * This function creates a mask to filter the mesh points that have a Z value
#  * less than or equal to zero, and returns a new mesh containing only those
#  * points. It can also retain any associated data for further processing.
#  *
#  * @param mesh The input PyVista mesh to be filtered.
#  * @return pv.PolyData A new mesh with points that have Z <= 0.
#  */
def camera_threshold_up(mesh):
    # Create a boolean mask for points where Z <= 0
    z_values = mesh.points[:, 2]
    mask = z_values <= 0

    # Extract only the valid points
    filtered_points = mesh.points[mask]

    # Optionally retain associated data if necessary
    filtered_mesh = pv.PolyData(filtered_points)
    filtered_mesh["Elevation"] = filtered_mesh.points[:, 2]  # Use Z for the gradient

    return filtered_mesh

# /**
#  * @brief Filters the mesh points where Z is greater than or equal to -4.5.
#  *
#  * This function creates a mask to filter the mesh points that have a Z value
#  * greater than or equal to -4.5, and returns a new mesh containing only those
#  * points. It can also retain any associated data for further processing.
#  *
#  * @param mesh The input PyVista mesh to be filtered.
#  * @return pv.PolyData A new mesh with points that have Z >= -4.5.
#  */
def camera_threshold_down(mesh, silo_object):
    # Check if the mesh is empty
    if mesh.n_points == 0:
        raise ValueError("La malla está vacía. No se puede procesar.")

    # Create a boolean mask for points where Z >= -4.5
    z_values = mesh.points[:, 2]
    mask = z_values >= - silo_object.alto_total

    # Extract only the valid points
    filtered_points = mesh.points[mask]

    # Optionally retain associated data if necessary
    filtered_mesh = pv.PolyData(filtered_points)
    filtered_mesh["Elevation"] = filtered_mesh.points[:, 2]  # Use Z for the gradient

    return filtered_mesh

# /**
#  * @brief Reduces the number of points in the mesh to a specified maximum.
#  *
#  * This function checks if the number of points in the input mesh exceeds the
#  * specified `max_points` parameter. If so, it randomly selects a subset of points
#  * to reduce the total number. The resulting mesh will contain no more than
#  * `max_points` points. If the number of points is already within the limit,
#  * the original mesh is returned.
#  *
#  * @param mesh The input PyVista mesh to be reduced.
#  * @param max_points The maximum number of points to retain in the mesh (default is 5000).
#  * @return pv.PolyData The mesh with reduced points, or the original mesh if it already fits the limit.
#  */
def reduce_points(mesh, max_points=5000):
    if mesh.n_points > max_points:
        indices = np.random.choice(mesh.n_points, max_points, replace=False)
        reduced_points = mesh.points[indices]
        reduced_points = pv.PolyData(reduced_points)
        reduced_points["Elevation"] = reduced_points.points[:, 2]
        return reduced_points
    return mesh


def volume_prism(silo_object, z_mean):
    """
    Calcula el volumen del pellet hasta una altura dada desde la base menor (b1).

    Parámetros:
    - b1, b2: bases del trapecio (b1 abajo, b2 arriba)
    - alto_prisma: altura total del prisma
    - ancho_prisma: ancho del prisma
    - z_mean: altura del pellet medida desde z=0) (siempre negativa)
    """
    z = (z_mean + silo_object.alto_ortoedro) + silo_object.alto_prisma
    # Base superior al nivel del pellet (interpolación lineal)
    b_h = silo_object.base_menor_prisma + (silo_object.base_mayor_prisma - silo_object.base_menor_prisma) * (z / silo_object.alto_prisma)

    # Área de la sección transversal del pellet (trapecio)
    area_pellet = (silo_object.base_menor_prisma + b_h) * z / 2

    # Volumen
    volumen = area_pellet * silo_object.ancho_prisma
    return volumen


def volume_pyramyd(silo_object, z, method='sigmoid'):
    h_pyramyd = silo_object.alto_total + z  # z siemnpre negativo

    if h_pyramyd <= 0:
        return 0

    # Geometría de la pirámide
    area1 = silo_object.largo * silo_object.ancho
    area2 = silo_object.largo_orificio * silo_object.ancho_orificio
    V_real = (1/3) * h_pyramyd * (area1 + area2 + np.sqrt(area1 * area2))

    z_empty = silo_object.z_vacio  # cuando el silo está vacío
    z_cuboid = - silo_object.alto_ortoedro

    if method == 'lineal':
        if z < z_empty:
            return 0
        elif z_empty <= z < z_cuboid:
            # Interpolación lineal
            factor = (z - z_empty) / (z_cuboid - z_empty)
            return V_real * factor
        else:
            return V_real

    elif method == 'sigmoid':
        z0 = (z_empty + z_cuboid)/2  # punto medio
        k = 10      # pendiente
        sigmoid_factor = 1 / (1 + np.exp(-k * (z - z0)))
        return V_real * sigmoid_factor

    else:
        raise ValueError(f"Método '{method}' no reconocido. Usa 'lineal' o 'sigmoid'.")



# /**
#  * @brief Measures the volume of a silo mesh.
#  *
#  * This function computes the volume of the silo mesh by first calculating the normals
#  * and then subtracting a predefined inverted pyramid volume. The resulting volume is
#  * rounded to the nearest integer for convenience.
#  *
#  * @param mesh_of_silo The input mesh representing the silo.
#  * @param volumen_silo_total The total volume of the silo (used for comparison).
#  * @return int The measured volume of the silo in cubic units (rounded).
#  */
def volume_measurement_of_silo(mesh_of_silo, z_mean, silo_object):
    z_cuboid = - silo_object.alto_ortoedro
    if z_mean > z_cuboid:
        mesh_of_silo = mesh_of_silo.compute_normals()
        measured_volume_of_silo = (mesh_of_silo.volume - silo_object.vol_compl_piram)
    else:
        measured_volume_of_silo = volume_pyramyd(silo_object, z_mean, method='sigmoid')
    if silo_object.silo_mitad:
        measured_volume_of_silo /= 2
    return round(measured_volume_of_silo)

def volume_measurement_of_silo_with_prism(mesh_of_silo, z_mean, silo_object):
    z_cuboid = - silo_object.alto_ortoedro
    z_prism = - silo_object.alto_prisma

    if z_mean > z_cuboid:
        mesh_of_silo = mesh_of_silo.compute_normals()
        measured_volume_of_silo = mesh_of_silo.volume - silo_object.vol_compl_piram
    elif (z_prism + z_cuboid) < z_mean <= z_cuboid:
        measured_volume_of_silo = volume_prism(silo_object, z_mean) + volume_pyramyd(silo_object, (z_cuboid + z_prism))
    elif z_mean <= (z_prism + z_cuboid):
        measured_volume_of_silo = volume_pyramyd(silo_object, z_mean, method='sigmoid')
    else:
        raise ValueError("z_mean está fuera de rango válido")

    return round(measured_volume_of_silo)

# /**
#  * @brief Extracts the surface from the given mesh.
#  *
#  * This function extracts the surface of a 3D mesh, simplifying the data by removing
#  * internal points and retaining only the surface geometry.
#  *
#  * @param mesh The input mesh from which the surface will be extracted.
#  * @return pv.PolyData The surface extracted from the input mesh.
#  */
def extract_surface(mesh):
    mesh = mesh.extract_surface()
    return mesh

# /**
#  * @brief Extracts and smoothes the surface of the given mesh.
#  *
#  * This function first extracts the geometry (surface) from the given mesh and then
#  * applies a Taubin smoothing filter to reduce noise and improve surface quality.
#  *
#  * @param mesh The input mesh from which the surface will be extracted and smoothed.
#  * @return pv.PolyData The smoothed surface extracted from the input mesh.
#  */
def extract_surface_smoothed(mesh):
    surface = mesh.extract_geometry()
    surface_smoothed = surface.smooth_taubin(n_iter=50, pass_band=0.05)
    return surface_smoothed

# /**
#  * @brief Processes the mesh by reducing points, extrapolating surface, extracting surface, and extruding it.
#  *
#  * This function first reduces the number of points in the mesh if it exceeds a given limit, then extrapolates the surface
#  * using a Random Forest model. Afterward, it extracts the surface, and finally, it extrudes the surface to create a 3D structure.
#  *
#  * @param mesh The input mesh that will undergo processing.
#  * @return pv.PolyData The final extruded surface after all processing steps.
#  */
def proceced_mesh(mesh):
    mesh_reduced = reduce_points(mesh)
    mesh_extrapoled = extrapolate_surface_rf(mesh_reduced, 5, 500)
    surface_extrapoled = extract_surface(mesh_extrapoled)
    surface_extruded = extrude_ply_surface(surface_extrapoled)
    return surface_extruded

# /**
#  * @brief Performs integrated mesh processing including thresholding, point reduction, polynomial and random forest extrapolation,
#  *        surface extraction, smoothing, and extrusion.
#  *
#  * This function applies a series of processing steps to the input mesh. It starts by applying a downward threshold,
#  * reducing points if necessary, and then performs two stages of polynomial extrapolation and one stage of random forest extrapolation.
#  * After that, it extracts the surface and applies smoothing to the surface. Finally, the surface is extruded.
#  *
#  * @param mesh The input mesh to be processed.
#  * @param degree_of_polynom_1 The polynomial degree for the first extrapolation.
#  * @param degree_of_polynom_2 The polynomial degree for the second extrapolation.
#  * @param limit_extrapolation_polynom_1 The range limit for the first polynomial extrapolation.
#  * @param limit_extrapolation_polynom_2 The range limit for the second polynomial extrapolation.
#  * @param limite_rf The range limit for the random forest extrapolation.
#  * @return pv.PolyData The final processed mesh after all steps.
#  */
def proceced_mesh_with_polynom_extrapolation(mesh, degree_of_polynom_1
                                     ,degree_of_polynom_2
                                     , limit_extrapolation_polynom_1
                                     ,limit_extrapolation_polynom_2
                                     , limite_rf):
    mesh_proceced = camera_threshold_down(mesh)
    mesh_proceced  = reduce_points(mesh_proceced ,5000)
    mesh_proceced  = extrapolate_polynomial_surface(mesh_proceced ,degree_of_polynom_1,limit_extrapolation_polynom_1,500)
    mesh_proceced  = extrapolate_surface_rf(mesh_proceced ,limite_rf,1000)
    mesh_proceced  = extrapolate_polynomial_surface(mesh_proceced ,degree_of_polynom_2,limit_extrapolation_polynom_2,500)
    mesh_proceced  = extract_surface_smoothed(mesh_proceced )
    mesh_proceced = extrude_ply_surface(mesh_proceced)
    return mesh_proceced

# /**
#  * @brief Measures the average distance between the points of two meshes.
#  *
#  * This function calculates the Euclidean distance between each point in the original mesh and the closest point in the extrapolated mesh
#  * using a KDTree for efficient nearest-neighbor search. It then computes and returns the average distance.
#  *
#  * @param mesh_extrapolada The extrapolated mesh used as a reference for distance measurement.
#  * @param mesh_original The original mesh whose points are to be compared to the extrapolated mesh.
#  * @return float The average distance between the two meshes.
#  */
def meassure_distance_between_surfaces(mesh_extrapoled, mesh_original):
    tree = KDTree(mesh_extrapoled.points)
    d_kdtree, idx = tree.query(mesh_original.points)
    mesh_original["distances"] = d_kdtree
    mean_distance_between_surfaces = np.mean(d_kdtree)
    return mean_distance_between_surfaces

# Función para truncar un número a un número específico de decimales
def truncate(number, decimals=5):
    factor = 10 ** decimals
    return int(number * factor) / factor
# /**
#  * @brief Calculates the average Z value of the mesh points.
#  *
#  * This function computes the average value of the Z-coordinate of all the points in the mesh.
#  * The mesh is assumed to be an instance of `pv.PolyData`.
#  *
#  * @param mesh The mesh whose points' average Z value will be calculated.
#  * @return float The average Z value of the mesh points.
#  * @throws TypeError If the input mesh is not an instance of pv.PolyData.
#  */
def find_mean_of_mesh(mesh):
    if not isinstance(mesh, pv.PolyData):
        raise TypeError("La malla debe ser una instancia de pv.PolyData")
    points = mesh.points
    value_mean_z = np.mean(points[:, 2])
    return truncate(value_mean_z, 5)

# /**
#  * @brief Finds the Z-coordinate of the point closest to the origin in the XY plane.
#  *
#  * This function calculates the Euclidean distance of each point in the mesh to the origin (0, 0) in the XY plane
#  * and returns the Z-coordinate of the point that is closest to the origin.
#  * The mesh is assumed to be an instance of `pv.PolyData`.
#  *
#  * @param mesh The mesh from which the Z-coordinate of the point closest to the origin will be retrieved.
#  * @return float The Z-coordinate of the point closest to the origin in the XY plane.
#  * @throws TypeError If the input mesh is not an instance of pv.PolyData.
#  */
def find_z_value_in_origin(mesh):
    if not isinstance(mesh, pv.PolyData):
        raise TypeError("La malla debe ser una instancia de pv.PolyData")

    points = mesh.points

    # Calcular la distancia XY de cada punto al origen (0, 0)
    distances = np.linalg.norm(points[:, :2], axis=1)

    # Índice del punto más cercano a (0, 0) en XY
    near_index = np.argmin(distances)

    # Retornar el valor de Z correspondiente a ese punto
    return points[near_index][2]

def proccesing_mesh_with_z_mean(mesh, silo_object):

    mesh =camera_threshold_down(mesh,silo_object)
    mesh = reduce_points(mesh)
    mean_of_z = find_mean_of_mesh(mesh)
    plane_with_z_mean = create_plane_ply(mean_of_z,silo_object)
    surface = extract_surface(plane_with_z_mean)
    extruded_surface= extrude_ply_surface(surface, silo_object)
    return extruded_surface , mean_of_z

def proccesing_mesh_with_1_point(mesh, silo_object):

    mesh =camera_threshold_down(mesh,silo_object)
    mesh = reduce_points(mesh)
    z_value_in_origin= find_z_value_in_origin(mesh)
    plane_with_z_mean = create_plane_ply(z_value_in_origin,silo_object)
    surface = extract_surface(plane_with_z_mean)
    extruded_surface= extrude_ply_surface(surface, silo_object)
    return extruded_surface

def proccesing_mesh_with_random_forest(mesh, silo_object):

    mesh =camera_threshold_down(mesh,silo_object)
    mesh = reduce_points(mesh)
    mesh_extrapolated = extrapolate_surface_rf(mesh, silo_object,1000)
    surface = extract_surface(mesh_extrapolated)
    extruded_surface= extrude_ply_surface(surface, silo_object)
    return extruded_surface