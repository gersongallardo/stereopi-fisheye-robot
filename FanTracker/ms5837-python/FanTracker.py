#!/usr/bin/python
import ms5837
import time
from datetime import datetime

sensor = ms5837.MS5837_30BA() # Default I2C bus is 1 (Raspberry Pi 3)

# We must initialize the sensor before reading it
if not sensor.init():
    print("Sensor could not be initialized")
    exit(1)

# Initial read to verify sensor is working
if not sensor.read():
    print("Sensor read failed!")
    exit(1)

print("=== Configuración inicial ===")
print(("Presión: %.2f atm  %.2f Torr  %.2f psi") % (
    sensor.pressure(ms5837.UNITS_atm),
    sensor.pressure(ms5837.UNITS_Torr),
    sensor.pressure(ms5837.UNITS_psi)))

print(("Temperatura: %.2f C  %.2f F  %.2f K") % (
    sensor.temperature(ms5837.UNITS_Centigrade),
    sensor.temperature(ms5837.UNITS_Farenheit),
    sensor.temperature(ms5837.UNITS_Kelvin)))

freshwaterDepth = sensor.depth()
sensor.setFluidDensity(ms5837.DENSITY_SALTWATER)
saltwaterDepth = sensor.depth()
sensor.setFluidDensity(1000) # kg/m^3
print(("Profundidad: %.3f m (agua dulce)  %.3f m (agua salada)") % (freshwaterDepth, saltwaterDepth))

print(("Altura relativa MSL: %.2f m") % sensor.altitude())
print("\n=== Iniciando mediciones cada 10 segundos ===\n")

# Mediciones continuas cada 10 segundos
while True:
    if sensor.read():
        # Obtener fecha y hora actual
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        
        # Obtener todos los datos
        pressure_mbar = sensor.pressure()
        temp_c = sensor.temperature()
        depth = sensor.depth()  # Profundidad en metros (agua dulce por defecto)
        altitude = sensor.altitude()
        
        # Mostrar medición completa
        print("[%s]  P: %0.1f mbar  |  Prof: %.3f m  |  T: %0.2f C  |  Altura: %.2f m" % (
            timestamp,
            pressure_mbar,
            depth,
            temp_c,
            altitude))
    else:
        print("Error: Lectura del sensor falló!")
        exit(1)
    
    # Esperar 10 segundos antes de la siguiente medición
    time.sleep(10)
