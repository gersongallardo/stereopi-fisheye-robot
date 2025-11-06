import csv
import os
from datetime import datetime
from logging import Logger

class ExporterData:
    def __init__(self, directory, csv_name, node, logger: Logger):
        """
        Inicializa el logger para trabajar con archivos CSV y .bin.
        Crea una carpeta y asigna un nombre único al archivo según la fecha y hora.
        :param directory: Directorio donde se guardarán los archivos.
        :param node: Nombre del nodo para los archivos generados.
        """
        self.directory = directory
        self.logger = logger
        self.node = node
        self.file_path = f'{directory}/{csv_name}_rpi{node}'
        self.csv_file = self.create_csv_file()

    def create_csv_file(self):
        """
        Crea un archivo CSV único en la carpeta con el nombre basado en la fecha y hora actuales.
        Si el archivo ya existe, lo reutiliza.
        :return: Ruta completa del archivo CSV.
        """
        csv_file_path = f'{self.file_path}.csv'
        
        if not os.path.exists(csv_file_path):
            # Crear el archivo con el encabezado
            with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(["ID", "Timestamp", "Filename", "Temperature", "Humidity", "AC"])  # Encabezado

            self.logger.info(f"Archivo CSV creado: {csv_file_path}")
        
        return csv_file_path

    def get_last_id(self):
        """
        Obtiene el último ID registrado en el CSV.
        :return: Último ID registrado o 0 si el archivo está vacío.
        """
        if not os.path.exists(self.csv_file):
            return 0  # Si el archivo no existe, empezamos desde 0
        
        with open(self.csv_file, newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader, None)  # Saltar la cabecera
            ids = [int(row[0]) for row in reader if row]  # Obtener la columna de IDs
        
        return max(ids) if ids else 0  # Devolver el máximo ID encontrado
    
    
    def save_data_sensors(self, data):
        """
        Almacena los valores de temperatura, humedad y AC_OK.
        """
        temperature = None
        humidity = None
        ac_ok = None
        
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")  # Obtener hora de muestra
        self.new_id = self.get_last_id() + 1  # Obtener el ID incrementado
        
        # Nombre de los archivos
        self.filename = f'{self.node}_{self.timestamp}'

        # Buscar dinámicamente el sensor que contiene temperatura y humedad
        for sensor_name, sensor_data in data.items():
            if 'temperature_c' in sensor_data and 'humidity' in sensor_data:
                temperature = sensor_data['temperature_c']
                humidity = sensor_data['humidity']
            if 'state' in sensor_data:  # Buscar el sensor que contiene el estado de AC_OK
                ac_ok = sensor_data['state']

        # Guardar los datos en el CSV con ID incremental
        with open(self.csv_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([self.new_id, self.timestamp, self.filename, temperature, humidity, ac_ok])

        self.logger.info("Datos guardados en CSV.")
        
        

    def save_matrix(self, depth_frame, ir_frame):
        """
        Guarda la matriz de profundidad y de imagen infrarroja en archivos .bin
        """
        self.logger.info(f"Matriz guardada: ID = {self.new_id}, Timestamp = {self.timestamp}")
        
        # Guardar la matriz de profundidad
        with open(f'{self.directory}/{self.filename}_df.bin', "wb") as f:
            f.write(depth_frame)
        self.logger.info(f"Matriz de profundidad guardada: {self.filename}_df.bin")

        if ir_frame is not None:
            # Guardar la matriz de imagen infrarroja
            with open(f'{self.directory}/{self.filename}_irf.bin', "wb") as f:
                f.write(ir_frame.get_data())
            self.logger.info(f"Matriz de imagen infrarroja guardada: {self.filename}_irf.bin")