#!/bin/bash

# Verifica las variables de entorno requeridas
if [ -z "$REMOTE_USER" ] || [ -z "$REMOTE_HOST" ] || [ -z "$REMOTE_PATH" ]; then
    echo "Error: Variables de entorno requeridas no están configuradas"
    echo "Por favor configura: REMOTE_USER, REMOTE_HOST, REMOTE_PATH"
    exit 1
fi

# Ruta local del proyecto
SOURCE_PATH="/home/pi3/projects_innovex/silo_cam"

# Log de timestamp
echo "Iniciando sincronización: $(date)"

# Sincronizar directorio out
echo "Sincronizando directorio out..."
rsync -avz --progress "${SOURCE_PATH}/out/" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/out/"

# Sincronizar directorio logs
echo "Sincronizando directorio logs..."
rsync -avz --progress "${SOURCE_PATH}/logs/" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/logs/"

echo "Sincronización completada: $(date)"