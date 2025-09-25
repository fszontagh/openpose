#!/bin/bash
set -e

BASE_DIR="/usr/share/openpose-cli/models"

declare -A urls=(
  ["$BASE_DIR/face/haarcascade_frontalface_alt.xml"]="https://huggingface.co/fszontagh/openpose-pose-estimation/resolve/main/face/haarcascade_frontalface_alt.xml"
  ["$BASE_DIR/face/pose_deploy.prototxt"]="https://huggingface.co/fszontagh/openpose-pose-estimation/resolve/main/face/pose_deploy.prototxt"
  ["$BASE_DIR/face/pose_iter_116000.caffemodel"]="https://huggingface.co/fszontagh/openpose-pose-estimation/resolve/main/face/pose_iter_116000.caffemodel"
  ["$BASE_DIR/hand/pose_deploy.prototxt"]="https://huggingface.co/fszontagh/openpose-pose-estimation/resolve/main/hand/pose_deploy.prototxt"
  ["$BASE_DIR/hand/pose_iter_102000.caffemodel"]="https://huggingface.co/fszontagh/openpose-pose-estimation/resolve/main/hand/pose_iter_102000.caffemodel"
  ["$BASE_DIR/person/MobileNetSSD_deploy.caffemodel"]="https://huggingface.co/fszontagh/openpose-pose-estimation/resolve/main/person/MobileNetSSD_deploy.caffemodel"
  ["$BASE_DIR/person/MobileNetSSD_deploy.prototxt"]="https://huggingface.co/fszontagh/openpose-pose-estimation/resolve/main/person/MobileNetSSD_deploy.prototxt"
  ["$BASE_DIR/pose/body_25/pose_deploy.prototxt"]="https://huggingface.co/fszontagh/openpose-pose-estimation/resolve/main/pose/body_25/pose_deploy.prototxt"
  ["$BASE_DIR/pose/body_25/pose_iter_584000.caffemodel"]="https://huggingface.co/fszontagh/openpose-pose-estimation/resolve/main/pose/body_25/pose_iter_584000.caffemodel"
  ["$BASE_DIR/pose/coco/pose_deploy_linevec.prototxt"]="https://huggingface.co/fszontagh/openpose-pose-estimation/resolve/main/pose/coco/pose_deploy_linevec.prototxt"
  ["$BASE_DIR/pose/coco/pose_iter_440000.caffemodel"]="https://huggingface.co/fszontagh/openpose-pose-estimation/resolve/main/pose/coco/pose_iter_440000.caffemodel"
  ["$BASE_DIR/pose/mpi/pose_deploy_linevec.prototxt"]="https://huggingface.co/fszontagh/openpose-pose-estimation/resolve/main/pose/mpi/pose_deploy_linevec.prototxt"
  ["$BASE_DIR/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"]="https://huggingface.co/fszontagh/openpose-pose-estimation/resolve/main/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
  ["$BASE_DIR/pose/mpi/pose_iter_160000.caffemodel"]="https://huggingface.co/fszontagh/openpose-pose-estimation/resolve/main/pose/mpi/pose_iter_160000.caffemodel"
)

for filepath in "${!urls[@]}"; do
  url=${urls[$filepath]}
  dir=$(dirname "$filepath")

  echo "Downloading $url to $filepath"
  mkdir -p "$dir"
  curl -L -o "$filepath" "$url"
done

echo "All models downloaded successfully."