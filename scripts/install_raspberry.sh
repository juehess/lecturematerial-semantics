#!/bin/bash

# Exit on error
set -e

echo "🚀 Starting installation for Raspberry Pi 4..."

# Function to print section headers
print_section() {
    echo -e "\n📦 $1..."
}

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/cpuinfo; then
    echo "⚠️  This script is intended for Raspberry Pi only!"
    exit 1
fi

# Update system
print_section "Updating system packages"
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies
print_section "Installing system dependencies"
sudo apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libjpeg-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libatlas-base-dev \
    python3-dev \
    python3-pip \
    python3-venv \
    libhdf5-dev \
    libhdf5-serial-dev \
    libjasper-dev \
    libqt4-test \
    libfontconfig1-dev \
    libcairo2-dev \
    libgdk-pixbuf2.0-dev \
    libpango1.0-dev \
    libgtk2.0-dev \
    libgtk-3-dev \
    libfreetype6-dev \
    libpng-dev \
    python3-numpy \
    wget \
    git

# Install miniforge if not already installed
print_section "Setting up miniforge"
if ! command -v conda &> /dev/null; then
    echo "Installing miniforge..."
    wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh -O miniforge.sh
    chmod +x miniforge.sh
    ./miniforge.sh -b
    rm miniforge.sh
    ~/miniforge3/bin/conda init
    source ~/.bashrc
else
    echo "Miniforge already installed"
fi

# Create and activate conda environment
print_section "Creating conda environment"
ENV_NAME="eah_segmentation"

# Remove existing environment if it exists
conda env remove -n $ENV_NAME -y || true

# Create new environment from environment_raspberry.yml
conda env create -f environment_raspberry.yml

echo -e "\n✅ Installation completed!"
echo "To activate the environment, run:"
echo "   conda activate $ENV_NAME"
echo -e "\n⚠️  Note: You may need to reboot your Raspberry Pi for all changes to take effect."
echo "After reboot, verify the installation by running the test notebook in notebooks/segmentation_example.ipynb" 