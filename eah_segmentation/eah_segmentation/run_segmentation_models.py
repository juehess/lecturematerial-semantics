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
    from .color_utils import colorize_mask
    from .model_inference import run_inference_on_image
except ImportError:
    from color_utils import colorize_mask
    from model_inference import run_inference_on_image

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
    'deeplabv3plus_xception': {
        'input_size': (512, 512),
        'preprocess': 'xception'
    },
    'segformer_b0': {
        'input_size': (512, 512),
        'preprocess': 'normalize'
    }
}

def preprocess_image(image_path, target_size, preprocess_type='normalize'):
    """Preprocess image for model input"""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize image
    img = cv2.resize(img, (target_size[1], target_size[0]))  # width, height
    
    # Apply model-specific preprocessing
    if preprocess_type == 'normalize':  # for SegFormer
        # Simple normalization to [0, 1]
        img = img.astype(np.float32) / 255.0
    elif preprocess_type in ('xception', 'mobilenet'):  # for DeepLab
        # Normalize to [-1, 1]
        img = img.astype(np.float32) / 127.5 - 1.0
    else:
        raise ValueError(f"Unknown preprocessing type: {preprocess_type}")
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

def load_model_from_path(model_name, model_dir):
    """Load model from local directory"""
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
    """Colorize the segmentation mask"""
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

def main():
    parser = argparse.ArgumentParser(description='Run segmentation models on an image')
    parser.add_argument('--input', type=str, required=True, help='Path to input image')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save results')
    parser.add_argument('--models', type=str, nargs='+', default=list(MODELS.keys()),
                      help='List of models to run (default: all models)')
    args = parser.parse_args()

    # Convert input path to absolute path
    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} does not exist")
        return

    # Create output directory if it doesn't exist
    output_dir = os.path.join(PROJECT_ROOT, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load and resize original image once
    original = cv2.imread(input_path)
    if original is None:
        print(f"Error: Could not read input image {input_path}")
        return

    # Process with each model
    for model_name in args.models:
        if model_name not in MODELS:
            print(f"Warning: Unknown model {model_name}, skipping...")
            continue

        print(f"\nProcessing with {model_name}...")
        model_info = MODELS[model_name]
        input_size = model_info['input_size']
        preprocess_type = model_info['preprocess']

        # Load and preprocess image
        try:
            image = preprocess_image(input_path, input_size, preprocess_type)
            print(f"Successfully loaded and preprocessed image with shape: {image.shape}")
            print(f"Preprocessed image min/max: {np.min(image)}/{np.max(image)}")
        except Exception as e:
            print(f"Error loading image: {str(e)}")
            continue

        # Load model
        model_dir = os.path.join(PROJECT_ROOT, 'models')
        try:
            model = load_model_from_path(model_name, model_dir)
            if model is None:
                print(f"Error loading model: {model_name}")
                continue
            print(f"Successfully loaded model")
        except Exception as e:
            print(f"Error loading model {model_name}: {str(e)}")
            continue

        try:
            predictions = run_inference_on_image(model, image, model_name, preprocess_type)
            if predictions is not None:
                print(f"Prediction shape: {predictions.shape}")
                
                # Save raw predictions
                output_path = os.path.join(output_dir, f"{model_name}_prediction.npy")
                np.save(output_path, predictions)
                print(f"Saved raw predictions to {output_path}")
                
                # Save colorized mask
                color_mask = colorize_mask(predictions)
                print(f"Color mask shape: {color_mask.shape}")
                mask_path = os.path.join(output_dir, f"{model_name}_mask.png")
                cv2.imwrite(mask_path, cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))
                print(f"Saved colorized mask to {mask_path}")
                
                # Create overlay
                # Resize original image to match mask size
                resized_original = cv2.resize(original, (input_size[1], input_size[0]))
                print(f"Resized original shape: {resized_original.shape}")
                
                # Ensure both images have the same dimensions
                if resized_original.shape[:2] != color_mask.shape[:2]:
                    print("Resizing color mask to match original image dimensions")
                    color_mask = cv2.resize(color_mask, (resized_original.shape[1], resized_original.shape[0]))
                
                # Convert color mask to BGR for overlay
                color_mask_bgr = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
                print(f"Color mask BGR shape: {color_mask_bgr.shape}")
                
                # Create overlay
                overlay = cv2.addWeighted(resized_original, 0.7, color_mask_bgr, 0.3, 0)
                overlay_path = os.path.join(output_dir, f"{model_name}_overlay.png")
                cv2.imwrite(overlay_path, overlay)
                print(f"Saved overlay to {overlay_path}")
        except Exception as e:
            print(f"Error processing {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
