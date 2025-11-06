#!/bin/bash

# Variables configurables
empresa="mw"
ubicacion="huarnorte"
silo="1"
sensor="1"
logfile="/home/pi/logsrsync/rsync_$(date +%Y%m%d).log"
directorio_tenten="nicolas.donoso@tenten.innovex.cl:/home/nicolas.donoso/silo_files/data/${empresa}/${ubicacion}"

# Ejecutar siempre el script Python
/usr/bin/python3 /home/pi/software/silo_cam/src/client/register_cam2.py >> /home/pi/software/silo_cam/logs.txt 2>&1

# Crear carpeta para logs del rsync
mkdir -p /home/pi/logsrsync

# Comprobar si hay otro rsync en curso
if pgrep -f "rsync.*/home/pi" > /dev/null; then
    echo "$(date) - Rsync ya está en ejecución. Saltando sincronización." >> "$logfile"
    exit 0
fi

# Ejecutar sincronización de logs
echo "$(date) - Iniciando sincronización de logs." >> "$logfile"
rsync -avz --progress --mkpath /home/pi/logs/ ${directorio_tenten}/logs/${silo}_${sensor} >> "$logfile" 2>&1

# Ejecutar sincronización de archivos .ply y .bin
echo "$(date) - Iniciando sincronización de archivos .ply y .bin." >> "$logfile"
rsync -avz --progress --mkpath /home/pi/Documents/ ${directorio_tenten}/${silo}_${sensor}/ >> "$logfile" 2>&1
echo "$(date) - Sincronización finalizada." >> "$logfile"
echo "" >> "$logfile"

# Limpiar logs de más de 7 días
#find /home/pi/logsrsync -type f -name "*.log" -mtime +7 -exec rm {} \;

# Elimina archivos .ply y .bin de mas de 50 días.
echo "$(date) - Eliminando archivos .ply y .bin con más de 50 días de antigüedad." >> "$logfile"
find /home/pi/Documents/ -type f \( -name "*.ply" -o -name "*.bin" \) -mtime +50 -print -delete >> "$logfile" 2>&1
echo "$(date) - Eliminación completada." >> "$logfile"
echo " " >> "$logfile"
