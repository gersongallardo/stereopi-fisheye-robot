#!/bin/bash

cd /home/pi3/projects_innovex/silo_cam

docker run --rm --privileged \
  -e NODE=0 \
  -e DATA_DIRECTORY_EXPORT=./out \
  -e CSV_NAME=data \
  -e IMAGE_RGB=True \
  -e TZ=$(cat /etc/timezone) \
  -v /etc/localtime:/etc/localtime:ro \
  -v ./src/:/app/src \
  -v ./out/:/app/out/ \
  -v ./logs/:/app/logs/ \
  pyrealsense2-image python3 src/client/register_cam.py