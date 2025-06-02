# EAH Semantic Segmentation

This package demonstrates semantic segmentation using multiple models (SegFormer, DeepLabV3+, and Mosaic) on the ADE20K dataset.

## Installation

### Prerequisites
- Python 3.7 or higher
- pip or conda

### Using pip
```bash
# Clone the repository
git clone https://github.com/yourusername/eah_segmentation.git
cd eah_segmentation

# Install the package
pip install -e .
```

### Using conda
```bash
# Clone the repository
git clone https://github.com/yourusername/eah_segmentation.git
cd eah_segmentation

# Create and activate conda environment
conda env create -f environment.yml
conda activate eah_segmentation
```

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

# Download models (this will be handled automatically by the package)
python -c "from eah_segmentation.model_download import download_models; download_models()"
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
├── eah_segmentation/
│   ├── __init__.py
│   ├── ade20k_utils.py
│   ├── model_download.py
│   ├── model_inference.py
│   └── test_models.py
├── models/
│   ├── segformer_b0/
│   ├── deeplabv3plus_edgetpu/
│   └── mosaic/
├── datasets/
│   └── ADE20K/
├── results/
├── requirements.txt
├── environment.yml
└── README.md
```

## Requirements
- tensorflow>=2.12.0
- tensorflow-hub>=0.13.0
- tensorflow-datasets>=4.9.0
- opencv-python>=4.7.0
- numpy>=1.23.0
- Pillow>=9.5.0
- kagglehub>=0.2.0
- transformers>=4.30.0

## License
This project is licensed under the MIT License - see the LICENSE file for details.
