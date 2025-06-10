import os
import shutil
import kagglehub
import tensorflow as tf
import numpy as np
from tensorflow import keras
import keras_hub
from pathlib import Path
import cv2

"""
eah_segmentation/download_models.py

This module handles the downloading and conversion of various segmentation models from Kaggle Hub.
It manages three types of models:
1. DeepLabV3+ EdgeTPU (both Keras and TFLite formats)
2. Mosaic (TFLite format)
3. SegFormer B0 (converts from Keras to TFLite format)

The module provides functionality to download pre-trained models, organize them in a consistent
directory structure, and perform model conversion with optimizations.

Constants:
    MODEL_PATHS (dict): Dictionary mapping model names to their Kaggle Hub paths for different formats
    BASE_DIR (str): Base directory path of the current module
    MODEL_DIR (str): Directory path where all models will be stored
"""

# Model aliases and their Kaggle Hub paths
MODEL_PATHS = {
    "deeplabv3plus_edgetpu": {
        "keras": "google/deeplab-edgetpu/tensorFlow2/default-argmax-m/1",
        "tflite": "google/deeplab-edgetpu/tfLite/default-argmax-m/1"
    },
    "mosaic": {
        "tflite": "google/mosaic/tfLite/mobilenetmultiavgseg/1"
    }
}

# Local base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

def download_model(name, model_paths, model_dir):
    """
    Downloads model files from Kaggle Hub and organizes them in a consistent directory structure.
    
    Args:
        name (str): Name of the model to download (e.g., 'deeplabv3plus_edgetpu', 'mosaic')
        model_paths (dict): Dictionary containing paths for different formats of the model
        model_dir (str): Base directory where models should be stored
    
    Key Operations:
        1. Creates model-specific directories
        2. Downloads each format (keras/tflite) from Kaggle Hub
        3. Organizes downloaded files into format-specific subdirectories
    
    Raises:
        Exception: If download or file organization fails
    """
    print(f"\n🔄 Downloading {name} from Kaggle Hub...")
    
    # Create model-specific directory
    model_specific_dir = os.path.join(model_dir, name)
    os.makedirs(model_specific_dir, exist_ok=True)
    
    try:
        # Download each available format
        for format_type, kaggle_path in model_paths.items():
            print(f"📥 Downloading {format_type} format...")
            model_src = kagglehub.model_download(kaggle_path)
            print(f"✅ Downloaded {format_type} to {model_src}")

            # Create format-specific directory
            format_dir = os.path.join(model_specific_dir, format_type)
            if os.path.exists(format_dir):
                print(f"✅ {format_type} already exists at {format_dir}, skipping copy.")
            else:
                shutil.copytree(model_src, format_dir)
                print(f"✅ Copied {format_type} to {format_dir}")
                
    except Exception as e:
        print(f"❌ Failed to process {name}: {e}")

def representative_dataset_gen():
    """
    Generates a representative dataset for model quantization.
    
    This generator function loads and preprocesses a small set of training images
    from the ADE20K dataset to use as calibration data for quantization.
    
    Key Operations:
        1. Loads first 10 images from ADE20K training set
        2. Resizes images to 512x512
        3. Converts to RGB and normalizes pixel values
        4. Adds batch dimension
    
    Yields:
        list: Contains a single preprocessed image tensor of shape [1, 512, 512, 3]
    """
    # Load a few images from the dataset
    dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'ADE20K')
    image_dir = os.path.join(dataset_path, 'images', 'training')
    
    # Get list of image files
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))][:10]
    
    for image_file in image_files:
        # Load and preprocess image
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)
        if image is None:
            continue
            
        # Resize to model input size
        image = cv2.resize(image, (512, 512))
        
        # Convert to RGB and normalize
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        yield [image]

def convert_segformer_to_tflite():
    """
    Converts the SegFormer B0 model from Keras format to optimized TFLite format.
    
    Key Operations:
        1. Downloads SegFormer B0 model (ADE20K preset) from Kaggle Hub
        2. Exports as TensorFlow SavedModel
        3. Converts to TFLite with following optimizations:
           - Default optimizations
           - INT8 quantization
           - Hardware acceleration support
           - Float16 support
           - Custom ops support
        4. Saves the optimized TFLite model
    
    Important Code Blocks:
        - Model optimization configuration:
            - Enables multiple optimization sets (TFLITE_BUILTINS, SELECT_TF_OPS)
            - Configures INT8 quantization with representative dataset
            - Enables float16 support for better performance
    
    Raises:
        Exception: If model conversion or saving fails
    """
    print("\n🚀 Starting SegFormer B0 conversion process...")
    print("📥 Loading SegFormer B0 model from KaggleHub (ADE20K preset)...")

    try:
        # Load model from KaggleHub
        model = keras_hub.models.SegFormerImageSegmenter.from_preset(
            "kaggle://keras/segformer/keras/segformer_b0_ade20k_512"
        )
        print("✅ Successfully loaded SegFormer model")

        # Create model-specific directory
        model_specific_dir = os.path.join(MODEL_DIR, "segformer_b0")
        os.makedirs(model_specific_dir, exist_ok=True)
        
        # Export model to SavedModel directory
        saved_model_path = os.path.join(model_specific_dir, "keras")
        print(f"💾 Exporting SavedModel to {saved_model_path}...")
        model.export(saved_model_path)
        print("✅ SavedModel exported successfully")
        
        # Convert to TFLite with optimizations
        print("🔄 Converting to TFLite format with optimizations...")
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
        
        # Enable optimizations
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Enable hardware acceleration and optimizations
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        
        # Enable INT8 quantization with representative dataset
        converter.target_spec.supported_ops.append(tf.lite.OpsSet.TFLITE_BUILTINS_INT8)
        converter.representative_dataset = representative_dataset_gen
        
        # Set optimization flags
        converter.target_spec.supported_types = [tf.float16]
        converter.allow_custom_ops = True
        
        # Convert the model
        tflite_model = converter.convert()
        print("✅ TFLite conversion completed")
        
        # Save TFLite model
        tflite_path = os.path.join(model_specific_dir, "tflite", "1.tflite")
        print(f"💾 Saving TFLite model to {tflite_path}...")
        os.makedirs(os.path.dirname(tflite_path), exist_ok=True)
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        print("✅ TFLite model saved successfully")
        print(f"📊 TFLite model size: {len(tflite_model) / (1024 * 1024):.2f} MB")

    except Exception as e:
        print(f"❌ Error during SegFormer conversion: {str(e)}")
        raise

def main():
    """
    Main entry point for the model download and conversion process.
    
    Key Operations:
        1. Creates necessary directories
        2. Downloads all models specified in MODEL_PATHS
        3. Performs SegFormer conversion
    
    The function orchestrates the entire process of downloading and organizing
    all required models for the segmentation pipeline.
    """
    print("🚀 Starting model downloads...")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Download models from Kaggle Hub
    for name, model_paths in MODEL_PATHS.items():
        download_model(name, model_paths, MODEL_DIR)

    # Download and convert SegFormer from Kaggle
    print("\n🚀 Downloading and converting SegFormer B0...")
    convert_segformer_to_tflite()

    print("\n🎉 All models downloaded and converted successfully.")

if __name__ == "__main__":
    main()
