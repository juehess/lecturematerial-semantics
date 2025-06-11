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

# System update and essential packages
print_section "Updating system packages"
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3.9-venv \
    python3-pip \
    git \
    wget \
    unzip \
    build-essential \
    cmake \
    pkg-config \
    libjpeg-dev \
    libtiff5-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libcanberra-gtk* \
    libatlas-base-dev \
    gfortran \
    python3-dev \
    gnupg

# Set Python 3.9 as the default python3
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# Install Miniconda
print_section "Installing Miniconda"
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda
rm ~/miniconda.sh
echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Add Coral repository and install Edge TPU runtime
print_section "Setting up Coral USB Accelerator"
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update

# Install Edge TPU runtime and PyCoral
sudo apt-get install -y libedgetpu1-std
sudo apt-get install -y python3-pycoral=2.0.0

# Create udev rules for Coral USB
print_section "Setting up udev rules for Coral USB"
echo 'SUBSYSTEM=="usb",ATTRS{idVendor}=="1a6e",ATTRS{idProduct}=="089a",MODE="0666",GROUP="plugdev"' | sudo tee /etc/udev/rules.d/99-coral-usb.rules
sudo udevadm control --reload-rules && sudo udevadm trigger

# System optimizations
print_section "Applying system optimizations"

# Increase swap size to 2GB
sudo dphys-swapfile swapoff
sudo sed -i 's/CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile
sudo dphys-swapfile setup
sudo dphys-swapfile swapon

# Disable GUI if not needed (uncomment if you want a headless setup)
# sudo systemctl set-default multi-user.target

# Add current user to plugdev group for Coral USB access
sudo usermod -aG plugdev $USER

# Optional: Disable GUI-related services to save resources
print_section "Optimizing system for headless operation"
if systemctl is-active --quiet lightdm; then
    echo "Disabling desktop environment (lightdm) on boot..."
    sudo systemctl disable lightdm
fi

# Print Coral installation status
print_section "Verifying Coral installation"
echo "Edge TPU runtime and PyCoral have been installed."
echo "⚠️  Important: After installation completes:"
echo "1. Reboot your Raspberry Pi"
echo "2. Unplug and replug your Coral USB accelerator"
echo "3. The LED on the Coral device should light up"

echo -e "\n✅ Installation completed!"
echo -e "\n📋 Next steps:"
echo "1. Reload your shell configuration:"
echo "   source ~/.bashrc"
echo "2. Activate the environment:"
echo "   conda activate eah_segmentation"
echo -e "\n⚠️  Note: You may need to reboot your Raspberry Pi for all changes to take effect."
echo "After reboot, verify the installation by running the test notebook in notebooks/segmentation_example.ipynb" 