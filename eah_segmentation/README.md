# EAH Semantic Segmentation

A demonstration of semantic segmentation with multiple models using TensorFlow, with support for deployment on embedded systems like Raspberry Pi and Coral EdgeTPU.

## Installation

The package can be installed in different ways depending on your use case:

### Local Development (with Jupyter Notebooks)

For local development, including running the Jupyter notebooks:

```bash
pip install -e ".[dev]"
```

This installs all development dependencies including Jupyter, matplotlib, and paramiko for SSH connections.

### Raspberry Pi Deployment

For deploying on a Raspberry Pi (without notebooks):

```bash
pip install -e ".[raspberry]"
```

This installs only the necessary dependencies for running inference on the Raspberry Pi.

### Coral EdgeTPU Support

If you're using a Coral EdgeTPU:

```bash
pip install -e ".[coral]"
```

This installs the TFLite runtime and Coral EdgeTPU support.

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

### Local Development

1. Start Jupyter:
```bash
jupyter notebook
```

2. Open either:
   - `notebooks/presentation_demo.ipynb` for demonstration
   - `notebooks/student_practice.ipynb` for practice exercises

### Raspberry Pi Deployment

1. Copy the `infer.py` script to your Raspberry Pi
2. Install the package with Raspberry Pi dependencies
3. Run inference:
```bash
python infer.py --model model.tflite --input image.jpg --output out.png --log metrics.json
```

### Testing Models
To test the models on the ADE20K dataset:

```bash
# Test all models
python -m eah_segmentation.test_models --models segformer_b0 deeplabv3plus_edgetpu mosaic --num_images 5

# Test specific model
python -m eah_segmentation.test_models --models segformer_b0 --num_images 1

# Test with TFLite models
python -m eah_segmentation.test_models --models segformer_b0 deeplabv3plus_edgetpu mosaic --model_type tflite --num_images 5
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
│   ├── presentation_demo.ipynb
│   ├── student_practice.ipynb
│   └── infer.py           # Inference script for Raspberry Pi
├── models/                # Model files
├── data/                  # Example images
└── results/              # Output directory
```

## Requirements

- Python 3.7 or higher
- For local development: See `[project.optional-dependencies.dev]` in pyproject.toml
- For Raspberry Pi: See `[project.optional-dependencies.raspberry]` in pyproject.toml
- For Coral EdgeTPU: See `[project.optional-dependencies.coral]` in pyproject.toml

## License

This project is licensed under the MIT License - see the LICENSE file for details.
