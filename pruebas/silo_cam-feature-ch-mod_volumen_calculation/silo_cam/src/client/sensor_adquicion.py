import time
import logging
from typing import Dict, Any, Optional
import adafruit_dht
import digitalio

class SensorCollector:
    """
    Clase para recolectar datos de múltiples sensores en una Raspberry Pi.
    Soporta sensores DHT22 y puede ser extendido para más sensores.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Inicializa el colector de sensores.
        
        Args:
            logger: Logger configurado para registrar eventos y datos
        """
        self.logger = logger
        self.sensors: Dict[str, Dict[str, Any]] = {}
        
    def add_dht22_sensor(self, name: str, pin) -> None:
        """
        Agrega un sensor DHT22 a la colección.
        
        Args:
            name: Identificador único del sensor
            pin: Pin GPIO de board donde está conectado el sensor (ej: board.D4)
        """
        try:
            dht_device = adafruit_dht.DHT22(pin, use_pulseio=False)
            self.sensors[name] = {
                'type': 'DHT22',
                'pin': pin,
                'device': dht_device
            }
            self.logger.info(f"Sensor DHT22 agregado en pin {pin} con nombre {name}")
        except Exception as e:
            self.logger.error(f"Error al inicializar sensor DHT22 {name}: {str(e)}")
            raise
            
    def read_dht22(self, sensor_info: Dict[str, Any]) -> Dict[str, Optional[float]]:
        """
        Lee los datos de un sensor DHT22.
        
        Returns:
            Diccionario con temperatura y humedad
        """
        try:
            device = sensor_info['device']
            temperature = device.temperature
            humidity = device.humidity
            
            if temperature is not None:
                temperature = round(temperature, 2)
            else:
                temperature = None
                
            return {
                'temperature_c': temperature,
                'humidity': round(humidity, 2) if humidity is not None else None
            }
            
        except RuntimeError as e:
            # Errores comunes de lectura del sensor, no los tratamos como errores críticos
            self.logger.warning(f"Error temporal de lectura: {str(e)}")
            return {
                'temperature_c': None,
                'humidity': None
            }
              
    def add_digital_sensor(self, name: str, pin) -> None:
        """
        Agrega un sensor digital (entrada GPIO) a la colección.

        Args:
            name: Identificador único del sensor.
            pin: Pin GPIO de board donde está conectado el sensor (ej: board.D20).
        """
        try:
            digital_device = digitalio.DigitalInOut(pin)
            digital_device.direction = digitalio.Direction.INPUT

            self.sensors[name] = {
                'type': 'DIGITAL',
                'pin': pin,
                'device': digital_device
            }
            self.logger.info(f"Sensor digital agregado en pin {pin} con nombre {name}")

        except Exception as e:
            self.logger.error(f"Error al inicializar sensor digital {name}: {str(e)}")
            raise
    
    def read_digital_sensor(self, sensor_info: Dict[str, Any]) -> Dict[str, bool]:
        """
        Lee el estado de un sensor digital.

        Returns:
            Diccionario con el estado del sensor (True = Alto, False = Bajo)
        """
        try:
            device = sensor_info['device']
            return {'state': device.value}  # HIGH (True) = AC OK, LOW (False) = AC desconectada
        except Exception as e:
            self.logger.error(f"Error al leer sensor digital: {str(e)}")
            return {'state': None}

            
    def collect_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Recolecta datos de todos los sensores registrados.
        
        Returns:
            Diccionario con los datos de todos los sensores
        """
        data = {}
        for name, sensor_info in self.sensors.items():
            try:
                if sensor_info['type'] == 'DHT22':
                    time.sleep(2)  # Espera antes de leer el sensor DHT22
                    readings = self.read_dht22(sensor_info)
                elif sensor_info['type'] == 'DIGITAL':
                    readings = self.read_digital_sensor(sensor_info)
                else:
                    self.logger.error(f"Tipo de sensor desconocido: {sensor_info['type']}")
                    continue
                    
                data[name] = readings
                
                if all(v is not None for v in readings.values()):
                    self.logger.debug(f"Datos recolectados de {name}: {readings}")
                else:
                    self.logger.warning(f"Lectura parcial o nula de {name}: {readings}")
                
            except Exception as e:
                self.logger.error(f"Error al leer sensor {name}: {str(e)}")
                data[name] = {
                    'error': str(e)
                }
                
        return data

    def cleanup(self):
        """
        Limpia y libera los recursos de los sensores.
        Importante llamar a este método al terminar.
        """
        for sensor_info in self.sensors.values():
            if sensor_info['type'] == 'DHT22':
                try:
                    sensor_info['device'].exit()
                except:
                    pass