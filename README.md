# EAH Semantic Segmentation

A comprehensive demonstration of semantic segmentation with multiple models using TensorFlow, featuring support for deployment on embedded systems like Raspberry Pi and Coral EdgeTPU. This project is designed for both educational purposes and practical applications in computer vision.

## Features

- **Multiple Model Support**: Compare and evaluate different segmentation models
  - SegFormer-B0: State-of-the-art transformer-based model
  - DeepLabV3+ (EdgeTPU optimized): Efficient model for edge devices
  - Mosaic: Custom model for specific use cases

- **Interactive Learning**: Jupyter notebooks for hands-on experience
  - Presentation demos with visualizations
  - Student practice exercises
  - Real-time model evaluation

- **Edge Device Support**: Deploy models on Raspberry Pi with Coral EdgeTPU
  - Optimized inference
  - Remote execution capabilities
  - Performance benchmarking

## Quick Start Guide

### 1. Environment Setup

Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate eah_segmentation
```

### 2. Package Installation

Install the package with development dependencies:
```bash
pip install -e ".[dev]"
```

This installs all necessary packages including:
- Jupyter and JupyterLab
- TensorFlow and related libraries
- Visualization tools
- SSH utilities for remote execution

### 3. Download Resources

Set up required directories and download models and dataset:
```bash
# Create directories
mkdir -p models datasets

# Download pre-trained models
python -m eah_segmentation.download_models

# Download ADE20K dataset
python -m eah_segmentation.download_dataset
```

### 4. Launch Jupyter Environment

Start Jupyter Lab (recommended) or Jupyter Notebook:
```bash
# Start Jupyter Lab (recommended)
jupyter lab

# Or start Jupyter Notebook
jupyter notebook
```

Navigate to the `notebooks` directory to access:
- `presentation_demo.ipynb`: Interactive demonstrations and model comparisons
- `student_practice.ipynb`: Guided learning exercises

#### Accessing Jupyter Lab in Browser

1. When you start Jupyter Lab, it will display a URL in the terminal, typically:
   ```
   http://localhost:8888/lab?token=<your-token>
   ```

2. Open this URL in your web browser to access Jupyter Lab

3. If you're running Jupyter Lab on a remote machine (like Raspberry Pi):
   - Use the machine's IP address instead of localhost
   - Example: `http://192.168.1.100:8888/lab?token=<your-token>`
   - Make sure the port (8888) is not blocked by firewall

## Why Jupyter Lab?

Jupyter Lab is our recommended interface for several compelling reasons:

### Enhanced Development Experience
- Modern, integrated development environment
- File browser and terminal in one window
- Multiple notebook tabs and split views
- Real-time code execution and visualization

### Advanced Features
- Enhanced code completion and IntelliSense
- Rich text editing with Markdown support
- Interactive data visualization
- Integrated debugging tools
- Git integration

### Performance Benefits
- Better handling of large notebooks
- Improved memory management
- Faster startup times
- More responsive interface

## Working with Models

### Available Models

1. **SegFormer-B0**
   - Transformer-based architecture
   - Trained on ADE20K dataset
   - Excellent accuracy for general scene understanding
   - Suitable for high-performance applications

2. **DeepLabV3+ (EdgeTPU optimized)**
   - Optimized for edge devices
   - Trained on ADE20K dataset
   - Efficient inference
   - Coral EdgeTPU compatible

3. **DeepLabV3**
   - Standard version
   - Trained on Cityscapes dataset
   - Optimized for urban scene segmentation
   - Good for street scenes and urban environments

4. **Mosaic**
   - Custom implementation
   - Trained on Cityscapes dataset
   - Specialized for urban scenarios
   - Balanced performance for street scenes

### Model Evaluation

Test models on the ADE20K dataset:
```bash
# Evaluate multiple models
python -m eah_segmentation.evaluate --models segformer_b0 deeplabv3plus_edgetpu mosaic --num_images 5

# Single model evaluation
python -m eah_segmentation.evaluate --models segformer_b0 --num_images 1

# TFLite model evaluation
python -m eah_segmentation.evaluate --models segformer_b0 deeplabv3plus_edgetpu mosaic --model_type tflite --num_images 5
```

### Command Line Arguments
- `--models`: List of model names to test (required)
  - Available options: segformer_b0, deeplabv3plus_edgetpu, mosaic
  - Can specify multiple models separated by spaces
- `--model_type`: Type of model to use
  - Options: 'tflite' or 'keras' (default: 'keras')
  - Use 'tflite' for optimized edge deployment
- `--num_images`: Number of images to test on
  - Default: 1
  - Higher values for more comprehensive evaluation
- `--output_dir`: Directory to save results
  - Default: 'results'
  - Creates model-specific subdirectories
- `--data_dir`: Directory containing ADE20K dataset
  - Default: 'datasets'
  - Must contain the ADE20K validation set
- `--image_index`: Index of the image to test
  - Default: 0
  - Useful for testing specific images

## Raspberry Pi Deployment

### Setup Instructions

1. Create and activate the Raspberry Pi environment:
```bash
conda env create -f environment_raspberry.yml
conda activate eah_segmentation_raspberry
pip install -e .
```

2. Find your Raspberry Pi's IP address:
```bash
# On Raspberry Pi, run:
hostname -I

# Or check your router's admin interface
# The IP will be something like 192.168.1.100
```

3. Connect to Raspberry Pi via SSH:
```bash
# Default username is 'pi'
ssh pi@192.168.1.100  # Replace with your Raspberry Pi's IP

# If this is your first time connecting, you'll see a fingerprint warning
# Type 'yes' to continue

# Enter the default password when prompted
# Default password is 'raspberry' (change this after first login)
```

4. Change the default password (recommended):
```bash
# After logging in, run:
passwd

# Enter the current password (raspberry)
# Enter your new password twice
```

5. Configure SSH connection in the notebooks:
```python
from eah_segmentation.ssh_utils import setup_ssh

# Configure SSH connection
ssh_config = {
    'hostname': '192.168.1.100',  # Replace with your Raspberry Pi's IP
    'username': 'pi',
    'password': 'your_new_password'  # Use your new password
}
ssh_client = setup_ssh(ssh_config)
```

### Performance Considerations
- Optimize model size for edge deployment
- Monitor memory usage
- Consider batch processing for efficiency
- Use TFLite models for better performance

## Project Structure

```
eah_segmentation/
├── notebooks/              # Jupyter notebooks
│   ├── presentation_demo.ipynb  # Interactive demonstrations
│   └── student_practice.ipynb   # Learning exercises
├── models/                # Model files and weights
├── data/                  # Example images and test data
└── results/              # Output directory for evaluations
```

## Requirements

### System Requirements
- Python 3.9 or higher
- Sufficient disk space for models and dataset
- GPU recommended for training (optional)

### Development Dependencies
See `[project.optional-dependencies.dev]` in pyproject.toml for full list:
- Jupyter and JupyterLab
- TensorFlow
- Matplotlib
- Paramiko (for SSH)

### Raspberry Pi Requirements
See environment_raspberry.yml for specific dependencies:
- TensorFlow Lite
- EdgeTPU runtime
- Minimal dependencies for inference

## Contributing

We welcome contributions! Please feel free to submit pull requests or open issues for:
- Bug reports
- Feature requests
- Documentation improvements
- Performance optimizations

## License

This project is licensed under the MIT License - see the LICENSE file for details.
