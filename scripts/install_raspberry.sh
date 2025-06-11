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
    python3-dev \
    python3-pip \
    python3-venv \
    python3-numpy \
    libatlas-base-dev \
    libhdf5-dev \
    libopenjp2-7-dev \
    libtbb-dev \
    libprotobuf-dev \
    protobuf-compiler \
    wget \
    git

# Install Coral USB dependencies
print_section "Installing Coral USB dependencies"
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install -y libedgetpu1-std
# Note: Use max frequency version only if you have good cooling
# sudo apt-get install -y libedgetpu1-max

# Add current user to plugdev group for Coral USB access
sudo usermod -aG plugdev $USER

# Create udev rules for Coral USB
print_section "Setting up Coral USB rules"
echo 'SUBSYSTEM=="usb",ATTRS{idVendor}=="1a6e",ATTRS{idProduct}=="089a",MODE="0666",GROUP="plugdev"' | sudo tee /etc/udev/rules.d/99-coral-edgetpu.rules
sudo udevadm control --reload-rules && sudo udevadm trigger

# Optional: Disable GUI-related services to save resources
print_section "Optimizing system for headless operation"
if systemctl is-active --quiet lightdm; then
    echo "Disabling desktop environment (lightdm) on boot..."
    sudo systemctl disable lightdm
fi

# Increase swap size for better ML performance
print_section "Configuring system for ML workloads"
SWAP_SIZE="2G"
echo "Setting swap size to $SWAP_SIZE..."
sudo sed -i "s/CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/" /etc/dphys-swapfile
sudo /etc/init.d/dphys-swapfile restart

# Install miniforge if not already installed
print_section "Setting up miniforge"
if ! command -v conda &> /dev/null; then
    echo "Installing miniforge..."
    wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh -O miniforge.sh
    chmod +x miniforge.sh
    ./miniforge.sh -b -p $HOME/miniforge3
    rm miniforge.sh
    
    # Initialize conda in the current shell
    export PATH="$HOME/miniforge3/bin:$PATH"
    
    # Initialize conda in .bashrc
    $HOME/miniforge3/bin/conda init bash
    
    # Also initialize for zsh if it exists
    if [ -f "$HOME/.zshrc" ]; then
        $HOME/miniforge3/bin/conda init zsh
    fi
    
    # Ensure conda is available in current session
    eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
    
    echo "Conda has been installed and initialized"
    echo "Please run 'source ~/.bashrc' after the script finishes"
else
    echo "Miniforge already installed"
fi

# Create and activate conda environment
print_section "Creating conda environment"
ENV_NAME="eah_segmentation"

# Ensure we're using conda from miniforge
export PATH="$HOME/miniforge3/bin:$PATH"

# Remove existing environment if it exists
conda env remove -n $ENV_NAME -y || true

# Create new environment from environment_raspberry.yml
conda env create -f environment_raspberry.yml

echo -e "\n✅ Installation completed!"
echo -e "\n📋 Next steps:"
echo "1. Reload your shell configuration:"
echo "   source ~/.bashrc"
echo "2. Activate the environment:"
echo "   conda activate $ENV_NAME"
echo -e "\n⚠️  Note: You may need to reboot your Raspberry Pi for all changes to take effect."
echo "After reboot, verify the installation by running the test notebook in notebooks/segmentation_example.ipynb" 