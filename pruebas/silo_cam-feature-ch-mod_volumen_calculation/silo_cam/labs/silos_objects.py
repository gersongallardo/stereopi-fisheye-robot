from dataclasses import dataclass
from typing import List

@dataclass
class Silo:
    empresa: str
    ubicacion: str
    silo: str
    sensor: str
    ancho: float
    largo: float
    alto_total: float
    vol_compl_piram: float
    silo_mitad: bool
    alto_piramide: float
    alto_prisma: float
    base_mayor_prisma: float
    base_menor_prisma: float
    ancho_prisma: float
    ancho_orificio: float
    largo_orificio: float
    alto_ortoedro: float
    volumen_maximo: float
    volumen_minimo: float
    z_vacio: float
    traslacion: List[float]
    rotacion: List[float]
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

def crear_silo(config: dict) -> Silo:
    ancho = config['ancho']
    largo = config['largo']
    alto_total = config['alto_total']

    return Silo(
        empresa=config['empresa'],
        ubicacion=config['ubicacion'],
        silo=config['silo'],
        sensor=config['sensor'],
        ancho=ancho,
        largo=largo,
        alto_total=alto_total,
        vol_compl_piram=config['vol_compl_piram'],
        silo_mitad=config.get('silo_mitad', False),
        alto_prisma=config.get('alto_prisma', None),
        alto_piramide=config['alto_piramide'],
        base_mayor_prisma=config.get('base_mayor_prisma', None),
        base_menor_prisma=config.get('base_menor_prisma', None),
        ancho_prisma=config.get('ancho_prisma', None),
        ancho_orificio=config['ancho_orificio'],
        largo_orificio=config['largo_orificio'],
        alto_ortoedro=config['alto_ortoedro'],
        volumen_maximo=config.get('volumen_maximo', None),
        volumen_minimo=config.get('volumen_minimo', 0),
        z_vacio=config['z_vacio'],
        traslacion=config['traslacion'],
        rotacion=config['rotacion'],
        x_min=config.get('x_min', -ancho/2),
        x_max=config.get('x_max', ancho/2),
        y_min=config.get('y_min', -largo/2),
        y_max=config.get('y_max', largo/2),
        z_min=config.get('z_min', -alto_total),
        z_max=config.get('z_max', 0),
    )

def silo_abtao(silo_id: str, z_vacio: float, volumen_maximo: float, traslacion: List[float], rotacion: List[float]) -> Silo:
    config = {
        'empresa': 'mef',
        'ubicacion': 'abtao',
        'silo': silo_id,
        'sensor': '1',
        'ancho': 5,
        'largo': 5,
        'alto_total': 4.65,
        'alto_ortoedro': 2.64,
        'alto_piramide': 2.01,
        'vol_compl_piram': 32.62,
        'ancho_orificio': 0.25,
        'largo_orificio': 0.25,
        'volumen_maximo': volumen_maximo,
        'z_vacio': z_vacio,
        'traslacion': traslacion,
        'rotacion': rotacion
    }
    return crear_silo(config)

def silo_chidhuapi3(silo_id: str, sensor: str, z_vacio: float, volumen_maximo: float, traslacion: List[float], rotacion: List[float], limits: List[float]) -> Silo:
    config = {
        'empresa': 'ce',
        'ubicacion': 'chidhuapi3',
        'silo': silo_id,
        'sensor': sensor,
        'ancho': 4.474,
        'largo': 4.474,
        'alto_total': 4.915,
        'alto_ortoedro': 3.56,
        'alto_piramide': 1.355,
        'vol_compl_piram': 7.48*2,
        'silo_mitad': True,
        'ancho_orificio': 1.22,
        'largo_orificio': 1.22,
        'volumen_maximo': volumen_maximo,
        'z_vacio': z_vacio,
        'traslacion': traslacion,
        'rotacion': rotacion,
        'x_min': limits[0],
        'x_max': limits[1],
        'y_min': limits[2],
        'y_max': limits[3]
    }
    return crear_silo(config)



def silo_huarnorte(silo_id: str, sensor: str, z_vacio: float, volumen_maximo: float, traslacion: List[float], rotacion: List[float]) -> Silo:
    config = {
        'empresa': 'mw',
        'ubicacion': 'huarnorte',
        'silo': silo_id,
        'sensor': sensor,
        'ancho': 6.5,
        'largo': 3.5,
        'alto_total': 5.5,
        'alto_ortoedro': 3.7,
        'alto_prisma': 1.12,
        'base_mayor_prisma': 6.5,
        'base_menor_prisma': 3.5,
        'ancho_prisma': 3.65,
        'alto_piramide': 0.68,
        'vol_compl_piram': 17.96,
        'ancho_orificio': 0.65,
        'largo_orificio': 0.65,
        'volumen_maximo': volumen_maximo,
        'z_vacio': z_vacio,
        'traslacion': traslacion,
        'rotacion': rotacion
    }
    return crear_silo(config)

# Definicion de silos
#abtao_1_1 = silo_abtao(silo_id='1', z_vacio = -3.63, volumen_maximo = 70,
#                       traslacion = [0.3, -0.1, 0], rotacion = [0, 0, 0])

abtao_2_1 = silo_abtao(silo_id='2', z_vacio = -3.63, volumen_maximo = 78.36,
                       traslacion = [1, -1, 0], rotacion = [0, 0, 0])

abtao_3_1 = silo_abtao(silo_id='3', z_vacio = -3.63, volumen_maximo = 78.36,
                       traslacion = [0.1, -0.3, 0], rotacion = [0, 0, 0])

abtao_4_1 = silo_abtao(silo_id='4', z_vacio = -3.63, volumen_maximo = 78.36,
                       traslacion = [-0.4, -0.35, 0], rotacion = [0, 0, 0])

#chidhuapi3_1_1 = silo_chidhuapi3(silo_id='1', sensor='1',z_vacio = -3.9, volumen_maximo = 38,
#                                 traslacion = [-0.27, -0.6, 0], rotacion = [0, 0, 5],
#                                 limits=[-4.474/2, 4.474/2, -4.474/2, 0])
#
#huarnorte_1_1 = silo_huarnorte(silo_id='1', sensor='1', z_vacio = -4.9, volumen_maximo = 50,
#                                    traslacion = [0.1, -0.29, 0], rotacion = [0, 0, 0])

# TODO ORDENAR DENSISDAD POR MARCA
DENSIDAD_PELLET_CHIDHUAPI            = 0.734





def silo_test(silo_id: str, z_vacio: float, volumen_maximo: float, traslacion: List[float], rotacion: List[float]) -> Silo:
    config = {
        'empresa': 'test',
        'ubicacion': 'test',
        'silo': silo_id,
        'sensor': '1',
        'ancho': 0.55,
        'largo': 0.55,
        'alto_total': 0.94,
        'alto_ortoedro': 0.57,
        'alto_piramide': 0.33,
        'vol_compl_piram': 0.04,
        'ancho_orificio': 0.25,
        'largo_orificio': 0.25,
        'volumen_maximo': volumen_maximo,
        'z_vacio': z_vacio,
        'traslacion': traslacion,
        'rotacion': rotacion
    }
    return crear_silo(config)

#silo_test = silo_test(silo_id='1', z_vacio = 0, volumen_maximo = 0,
#                       traslacion = [0, 0, 0], rotacion = [0, 0, 0])