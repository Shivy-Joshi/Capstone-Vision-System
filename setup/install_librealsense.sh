#!/usr/bin/env bash
set -e

LRS_VERSION="v2.56.1"

echo "[1/7] Remove old librealsense folders..."
rm -rf ~/librealsense ~/librealsense_build

echo "[2/7] Clone librealsense..."
git clone --branch ${LRS_VERSION} --depth 1 https://github.com/IntelRealSense/librealsense.git ~/librealsense

echo "[3/7] Install udev rules..."
cd ~/librealsense
sudo cp config/99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger

echo "[4/7] Build librealsense with RSUSB backend and Python bindings..."
mkdir -p ~/librealsense_build
cd ~/librealsense_build

cmake ../librealsense \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_EXAMPLES=true \
    -DBUILD_GRAPHICAL_EXAMPLES=false \
    -DBUILD_PYTHON_BINDINGS=true \
    -DPYTHON_EXECUTABLE=$(which python3) \
    -DFORCE_RSUSB_BACKEND=true \
    -DBUILD_WITH_CUDA=false

echo "[5/7] Compile..."
make -j$(nproc)

echo "[6/7] Install..."
sudo make install
sudo ldconfig

echo "[7/7] Install Python wrapper into your venv if active..."
if [[ -n "$VIRTUAL_ENV" ]]; then
    PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    RS_SO=$(find . -name "pyrealsense2*.so" | head -n 1)

    if [[ -n "$RS_SO" ]]; then
        SITE_PACKAGES="$VIRTUAL_ENV/lib/python${PYVER}/site-packages"
        mkdir -p "$SITE_PACKAGES"
        cp "$RS_SO" "$SITE_PACKAGES/"
        echo "Copied pyrealsense2 to $SITE_PACKAGES"
    else
        echo "Could not find pyrealsense2 shared object after build."
    fi
else
    echo "No venv active. Activate your venv and re-copy pyrealsense2 manually if needed."
fi

echo "Testing camera detection:"
rs-enumerate-devices || true

echo "Done."