# EAH Semantic Segmentation

A demonstration of semantic segmentation with multiple models using TensorFlow, with support for deployment on embedded systems like Raspberry Pi and Coral EdgeTPU.

## Installation

The package can be installed in different ways depending on your use case:

### Local Development (with Jupyter Notebooks)

For local development, including running the Jupyter notebooks:

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate eah_segmentation

# Install package with development dependencies
pip install -e ".[dev]"
```

This installs all development dependencies including Jupyter, matplotlib, and paramiko for SSH connections.

### Raspberry Pi Deployment

For deploying on a Raspberry Pi with Coral EdgeTPU:

```bash
# Create and activate conda environment
conda env create -f environment_raspberry.yml
conda activate eah_segmentation_raspberry

# Install package
pip install -e .
```

This installs the necessary dependencies for running inference on the Raspberry Pi with Coral EdgeTPU support.

## Downloading Models and Dataset

### Models
The package supports three models:
1. SegFormer-B0
2. DeepLabV3+ (EdgeTPU optimized)
3. Mosaic

To download the models, run:
```bash
# Create models directory
mkdir -p models

# Download models
python -m eah_segmentation.download_models
```

### ADE20K Dataset
The ADE20K dataset is required for testing. You can download it using:
```bash
# Create datasets directory
mkdir -p datasets

# Download ADE20K dataset (this will be handled automatically by the package)
python -c "from eah_segmentation.ade20k_utils import download_ade20k; download_ade20k()"
```

## Usage

### Jupyter Notebooks

The project includes two Jupyter notebooks for different purposes:

1. `notebooks/presentation_demo.ipynb`:
   - Designed for presentations and demonstrations
   - Runs models and visualizes results
   - Includes performance metrics and comparisons
   - Can run inference on Raspberry Pi via SSH

2. `notebooks/student_practice.ipynb`:
   - Designed for student practice sessions
   - Includes specific questions and tasks
   - Guides through model usage and evaluation
   - Helps understand semantic segmentation concepts

To start working with the notebooks:
```bash
# Start Jupyter
jupyter notebook

# Navigate to the notebooks directory
cd notebooks
```

### Raspberry Pi Deployment

The notebooks can run inference on a Raspberry Pi with Coral EdgeTPU via SSH. To set this up:

1. Ensure the Raspberry Pi is running and accessible via SSH
2. In the notebooks, configure the SSH connection:
   ```python
   from eah_segmentation.ssh_utils import setup_ssh
   
   # Configure SSH connection
   ssh_config = {
       'hostname': 'raspberry_pi_ip',
       'username': 'pi',
       'password': 'your_password'  # Or use key-based authentication
   }
   ssh_client = setup_ssh(ssh_config)
   ```

3. Use the provided functions to run inference remotely:
   ```python
   from eah_segmentation.inference import run_remote_inference
   
   # Run inference on Raspberry Pi
   results = run_remote_inference(ssh_client, model_name, image_path)
   ```

### Testing Models
To test the models on the ADE20K dataset:

```bash
# Run evaluation on multiple models
python -m eah_segmentation.evaluate --models segformer_b0 deeplabv3plus_edgetpu mosaic --num_images 5

# Run evaluation on a single model
python -m eah_segmentation.evaluate --models segformer_b0 --num_images 1

# Run evaluation using TFLite models
python -m eah_segmentation.evaluate --models segformer_b0 deeplabv3plus_edgetpu mosaic --model_type tflite --num_images 5
```

### Command Line Arguments
- `--models`: List of model names to test (required)
- `--model_type`: Type of model to use ('tflite' or 'keras', default: 'keras')
- `--num_images`: Number of images to test on (default: 1)
- `--output_dir`: Directory to save results (default: 'results')
- `--data_dir`: Directory containing ADE20K dataset (default: 'datasets')
- `--image_index`: Index of the image to test (default: 0)

### Results
For each processed image, a single visualization file is created in the model-specific output directory (e.g., `results/segformer_b0_keras/`). The visualization is a horizontal concatenation of three images:
- Left: Original RGB image
- Middle: Ground truth segmentation mask (colorized)
- Right: Predicted segmentation mask (colorized)

The files are named `prediction_XXXX.png` where XXXX is the image index.

## Project Structure

```
eah_segmentation/
├── notebooks/              # Jupyter notebooks
│   ├── presentation_demo.ipynb  # For demonstrations
│   └── student_practice.ipynb   # For student exercises
├── models/                # Model files
├── data/                  # Example images
└── results/              # Output directory
```

## Requirements

- Python 3.9 or higher
- For local development: See `[project.optional-dependencies.dev]` in pyproject.toml
- For Raspberry Pi: See environment_raspberry.yml

## License

This project is licensed under the MIT License - see the LICENSE file for details.
