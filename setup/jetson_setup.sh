#!/usr/bin/env bash
set -e

echo "[1/5] Updating apt..."
sudo apt update
sudo apt upgrade -y

echo "[2/5] Installing base packages..."
sudo apt install -y \
    git cmake build-essential pkg-config \
    python3-pip python3-dev python3-venv \
    libusb-1.0-0-dev libgtk-3-dev \
    libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev \
    libssl-dev libcurl4-openssl-dev \
    libjpeg-dev libtiff5-dev libpng-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libeigen3-dev \
    v4l-utils usbutils

echo "[3/5] Installing JetPack meta-package if not already present..."
sudo apt install -y nvidia-jetpack || true

echo "[4/5] Creating Python virtual environment..."
python3 -m venv ~/venvs/jetson-vision
source ~/venvs/jetson-vision/bin/activate
python -m pip install --upgrade pip wheel setuptools

echo "[5/5] Installing Python packages..."
pip install numpy opencv-python pupil-apriltags pyyaml

echo "Done."
echo "Activate later with:"
echo "source ~/venvs/jetson-vision/bin/activate"