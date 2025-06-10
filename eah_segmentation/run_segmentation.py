# eah_segmentation/run_segmentation_models.py

import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from pathlib import Path
import argparse

# Change relative imports to absolute imports
try:
    from .visualization import colorize_mask
    from .inference import run_inference_on_image
except ImportError:
    from visualization import colorize_mask
    from inference import run_inference_on_image

# Get the directory of the script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (one level up from script directory)
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# Add project root to Python path
sys.path.append(PROJECT_ROOT)

print(f"Script directory: {SCRIPT_DIR}")
print(f"Project root: {PROJECT_ROOT}")

# Define models with their configurations
MODELS = {
    'deeplabv3plus_edgetpu': {
        'input_size': (512, 512),
        'preprocess': 'mobilenet'
    },
    'segformer_b0': {
        'input_size': (512, 512),
        'preprocess': 'normalize'
    },
}

def load_model_from_path(model_name, model_dir):
    """
    Loads a model from local directory based on model name.
    
    Args:
        model_name (str): Name of the model to load
        model_dir (str): Base directory containing model files
        
    Returns:
        tf.keras.Model: Loaded model instance
    """
    try:
        # Map model names to their paths
        model_paths = {
            'deeplabv3plus_edgetpu': os.path.join(model_dir, 'default-argmax-m'),
            'deeplabv3': os.path.join(model_dir, 'default'),
            'segformer_b0': os.path.join(model_dir, 'segformer_b0')
        }
        
        # Get the path for this model
        model_path = model_paths.get(model_name)
        if not model_path:
            raise ValueError(f"Unknown model: {model_name}")
            
        if not os.path.exists(model_path):
            raise ValueError(f"Model not found: {model_path}")
        
        # Load the model
        model = tf.saved_model.load(model_path)
        print(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def colorize_mask(mask, num_classes=19):
    """
    Converts a segmentation mask to a colored visualization.
    
    Uses the Cityscapes color palette for visualization.
    
    Args:
        mask (np.ndarray): Segmentation mask
        num_classes (int): Number of classes in the mask
        
    Returns:
        np.ndarray: Colored visualization of the mask
    """
    # Cityscapes color palette
    palette = np.array([
        [128, 64, 128],   # road
        [244, 35, 232],   # sidewalk
        [70, 70, 70],     # building
        [102, 102, 156],  # wall
        [190, 153, 153],  # fence
        [153, 153, 153],  # pole
        [250, 170, 30],   # traffic light
        [220, 220, 0],    # traffic sign
        [107, 142, 35],   # vegetation
        [152, 251, 152],  # terrain
        [70, 130, 180],   # sky
        [220, 20, 60],    # person
        [255, 0, 0],      # rider
        [0, 0, 142],      # car
        [0, 0, 70],       # truck
        [0, 60, 100],     # bus
        [0, 80, 100],     # train
        [0, 0, 230],      # motorcycle
        [119, 11, 32]     # bicycle
    ])
    
    # Get the class with highest probability for each pixel
    if len(mask.shape) == 4:  # If we have a batch dimension
        mask = mask[0]  # Take first image in batch
    
    # Ensure mask is 2D
    if len(mask.shape) == 3:
        print(f"Mask shape before argmax: {mask.shape}")
        print(f"Mask min/max before argmax: {np.min(mask)}/{np.max(mask)}")
        class_mask = np.argmax(mask, axis=-1)
        print(f"Class mask unique values: {np.unique(class_mask)}")
    else:
        class_mask = mask
    
    # Create color mask
    color_mask = np.zeros((*class_mask.shape, 3), dtype=np.uint8)
    for i in range(num_classes):
        color_mask[class_mask == i] = palette[i]
    
    return color_mask

def run_inference_on_image(model, image, model_name, preprocess_type='normalize'):
    """Run inference on preprocessed image"""
    try:
        if model_name == 'deeplabv3':
            # TFLite inference
            input_details = model.get_input_details()
            output_details = model.get_output_details()
            
            # Set input tensor
            model.set_tensor(input_details[0]['index'], image)
            
            # Run inference
            model.invoke()
            
            # Get output tensor
            predictions = model.get_tensor(output_details[0]['index'])
        else:
            # SavedModel inference
            predictions = model(image)
            
        print(f"Raw prediction shape: {predictions.shape}")
        print(f"Raw prediction min/max: {np.min(predictions)}/{np.max(predictions)}")
        
        # Ensure predictions are in the correct shape (batch, height, width, classes)
        if len(predictions.shape) == 3:
            predictions = np.expand_dims(predictions, axis=0)
        
        # If predictions are logits, apply softmax
        if np.max(predictions) > 1.0:
            print("Applying softmax to logits")
            predictions = tf.nn.softmax(predictions, axis=-1).numpy()
        
        print(f"Processed prediction shape: {predictions.shape}")
        print(f"Processed prediction min/max: {np.min(predictions)}/{np.max(predictions)}")
        return predictions
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return None