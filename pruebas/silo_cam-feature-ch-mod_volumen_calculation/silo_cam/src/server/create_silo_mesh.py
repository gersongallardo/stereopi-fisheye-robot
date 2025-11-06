import trimesh
import numpy as np
import open3d as o3d
import silos_objects as silo


# Parámetros silo
silo = silo.abtao_2_1
#silo = silo.chidhuapi3_1_1
#silo = silo.huarnorte_1_1


# --- Crear cubo ---
cubo = trimesh.creation.box(extents=(silo.ancho, silo.largo, silo.alto_ortoedro))
# Posicionar para que su base esté en z = piramide_altura
cubo.apply_translation((silo.ancho / 2, silo.largo / 2, silo.alto_piramide + silo.alto_ortoedro / 2))

# --- Crear pirámide truncada invertida ---
def crear_piramide_truncada(ancho_silo, largo_silo, ancho_orificio, largo_orificio, alto_piramide):
    a_s = ancho_silo / 2
    l_s = largo_silo / 2
    a_o = ancho_orificio / 2
    l_o = largo_orificio / 2
    a_p = alto_piramide
    base = np.array([[-a_s, -l_s, 0], [a_s, -l_s, 0], [a_s, l_s, 0], [-a_s, l_s, 0]])
    top = np.array([[-a_o, -l_o, a_p], [a_o, -l_o, a_p], [a_o, l_o, a_p], [-a_o, l_o, a_p]])
    vertices = np.vstack((base, top))
    faces = [
        [2, 1, 0], [3, 2, 0],        # base
        [4, 5, 6], [4, 6, 7],        # top
        [0, 1, 5], [0, 5, 4],        # lado 1
        [1, 2, 6], [1, 6, 5],        # lado 2
        [2, 3, 7], [2, 7, 6],        # lado 3
        [3, 0, 4], [3, 4, 7]         # lado 4
    ]
    return trimesh.Trimesh(vertices=vertices, faces=faces)

piramide = crear_piramide_truncada(silo.ancho, silo.largo, silo.ancho_orificio, silo.largo_orificio, silo.alto_piramide)
piramide.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))  # invertir
# Base pequeña en z=0, base grande en z=piramide_altura
piramide.apply_translation([silo.ancho / 2, silo.largo / 2, silo.alto_piramide])


# --- Unir cubo y pirámide para crear silo cerrado ---
silo_completo = cubo.union(piramide)  # ambos son mallas cerradas



def cortar_mitad(mesh, eje='x', lado='izquierdo', tolerancia=1e-3):
    """
    Corta una malla por la mitad en el eje indicado, conservando el lado elegido.

    Parámetros:
        mesh: trimesh.Trimesh - malla original.
        eje: str - 'x' o 'y'.
        lado: str - 'izquierdo'/'derecho' para eje x, 'inferior'/'superior' para eje y.
        tolerancia: float - margen para evitar errores geométricos.

    Retorna:
        trimesh.Trimesh - malla recortada.
    """
    assert eje in ['x', 'y'], "El eje debe ser 'x' o 'y'"
    assert lado in ['izquierdo', 'derecho', 'inferior', 'superior'], \
        "El lado debe ser 'izquierdo', 'derecho', 'inferior' o 'superior'"

    axis_index = {'x': 0, 'y': 1}[eje]
    bounds = mesh.bounds
    centro = (bounds[0][axis_index] + bounds[1][axis_index]) / 2
    ancho = bounds[1][axis_index] - bounds[0][axis_index]

    # Crear caja más grande que el modelo en los otros ejes
    extents = [10, 10, 10]
    extents[axis_index] = ancho / 2 + tolerancia
    caja = trimesh.creation.box(extents=extents)

    # Posición de la caja depende del lado a conservar
    translation = [5, 5, 5]
    if eje == 'x':
        if lado == 'izquierdo':
            translation[0] = centro - ancho / 4
        elif lado == 'derecho':
            translation[0] = centro + ancho / 4
    elif eje == 'y':
        if lado == 'inferior':
            translation[1] = centro - ancho / 4
        elif lado == 'superior':
            translation[1] = centro + ancho / 4

    caja.apply_translation(translation)

    # Intersección
    resultado = mesh.intersection(caja)
    return resultado


# --- Quitar tapas para visualización/exportación ---
def quitar_tapas(mesh, eje='z', tolerancia=1e-5):
    axis_index = {'x': 0, 'y': 1, 'z': 2}[eje]
    verts = mesh.vertices
    faces = mesh.faces
    z_vals = verts[:, axis_index]
    z_min = z_vals.min()
    z_max = z_vals.max()

    def es_tapa(face):
        return all(abs(z_vals[v] - z_min) < tolerancia or abs(z_vals[v] - z_max) < tolerancia for v in face)

    caras_filtradas = np.array([face for face in faces if not es_tapa(face)])
    return trimesh.Trimesh(vertices=verts, faces=caras_filtradas)

# --- Visualizar en Open3D ---
def trimesh_to_open3d(mesh, color=[0.7, 0.5, 0.3]):
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    o3d_mesh.compute_vertex_normals()
    o3d_mesh.paint_uniform_color(color)
    return o3d_mesh


# --- (Opcional) Cortar mitad del silo ---
if silo.silo_mitad:
    silo_final = cortar_mitad(silo_completo, eje='y', lado='inferior')
else:
    silo_final = silo_completo

silo_sin_tapas = quitar_tapas(silo_final)

axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
o3d_mesh = trimesh_to_open3d(silo_sin_tapas)



def create_grid(size=5, step=1):
    lines = []
    points = []
    idx = 0
    for i in range(0, size + 1, step):  # Ahora la cuadrícula empieza desde (0, 0) hasta (size, size)
        # Líneas en la dirección X
        points.append([i, 0, 0])  # Comienza en 0,0
        points.append([i, size, 0])  # Llega hasta 5 en Y
        lines.append([idx, idx + 1])
        idx += 2

        # Líneas en la dirección Y
        points.append([0, i, 0])  # Comienza en 0,0
        points.append([size, i, 0])  # Llega hasta 5 en X
        lines.append([idx, idx + 1])
        idx += 2

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    return line_set

grid = create_grid(size=5, step=1)  # Crear una cuadrícula de 5x5 metros con paso de 1 metro


o3d.visualization.draw_geometries([o3d_mesh, axis, grid])

# --- Guardar resultado ---
silo_sin_tapas.export(f'./labs/silos_mesh/{silo.empresa}-{silo.ubicacion}.ply')
print("✅ Silo generado y guardado.")



# --- Crear cubo para restar a la piramide --
# Bounding box
extents = silo_final.extents  # [X, Y, Z] tamaño total en cada eje
ancho = extents[0]
largo = extents[1]

print(f"Ancho (X): {ancho}")
print(f"Largo (Y): {largo}")

cubo_mitad = trimesh.creation.box(extents=(ancho, largo, silo.alto_piramide))
cubo_mitad.apply_translation((ancho / 2, largo / 2, silo.alto_piramide/2))  # base en z=0

mesh_cubo_mitad = trimesh_to_open3d(cubo_mitad, color=[0.7, 0.7, 0.9])
o3d.visualization.draw_geometries([mesh_cubo_mitad, axis, grid])

# --- Resta booleana (cubo - pirámide) ---
piramide_restante = cubo_mitad.difference(silo_final)  # usar 'scad', 'igl' o 'blender' según disponibilidad
mesh_diferencia = trimesh_to_open3d(piramide_restante, color=[0.6, 0.9, 0.3])

# --- Visualizar resultado ---
o3d.visualization.draw_geometries([mesh_diferencia, axis, grid])

volumen_complemento_base = piramide_restante.volume
print(f"📐 Volumen del complemento de la base: {volumen_complemento_base:.2f} unidades cúbicas")

volumen_silo = silo_final.volume
print(f"📐 Volumen del silo: {volumen_silo:.2f} unidades cúbicas")