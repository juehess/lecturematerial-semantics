"""Model evaluation and metrics utilities."""

import time
import numpy as np
from pathlib import Path
import cv2

try:
    from inference import run_inference_on_image
    from ade20k_utils import save_prediction
except ImportError:
    from eah_segmentation.inference import run_inference_on_image
    from eah_segmentation.ade20k_utils import save_prediction

def evaluate_model(model, dataset, output_dir, num_images=10, model_name=None):
    """
    Evaluates a model on the ADE20K dataset and generates visualizations.
    
    Key Operations:
        - Runs inference on specified number of images
        - Measures and records inference times
        - Generates and saves visualizations
        - Computes timing statistics
    
    Args:
        model: The model to evaluate (TFLite or Keras)
        dataset: ADE20K dataset iterator
        output_dir (str): Directory to save results
        num_images (int): Number of images to evaluate
        model_name (str): Name of the model for logging
        
    Returns:
        dict: Dictionary containing timing statistics
    """
    print(f"\n🔍 Evaluating model on {num_images} image{'s' if num_images > 1 else ''}...")
    
    # Initialize timing statistics
    inference_times = []
    
    # Determine model type for visualization
    model_type = 'ade20k'
    if model_name:
        if 'cityscapes' in model_name.lower():
            model_type = 'cityscapes'
    
    for i, (image, true_mask) in enumerate(dataset):
        if i >= num_images:
            break
            
        # Run inference
        print(f"📸 Processing image {i+1}...")
        
        # Get ground truth classes
        true_classes = np.unique(true_mask.numpy()[0])
        print("\n📊 Ground truth classes:")
        for cls in true_classes:
            print(f"  Class {cls}")
        
        # Run inference and measure time
        pred_mask, inference_time = run_inference_on_image(model, image.numpy()[0], model_name, true_classes)
        inference_times.append(inference_time)
        
        print(f"⏱️  Inference time: {inference_time:.3f} seconds")
        
        # Save visualization
        save_prediction(
            image.numpy()[0],
            true_mask.numpy()[0],
            pred_mask,
            output_dir,
            i,
            model_type=model_type
        )
        
        print(f"✅ Processed image {i+1}/{num_images}")
    
    # Calculate and print timing statistics
    avg_time = np.mean(inference_times)
    std_time = np.std(inference_times)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)
    
    print(f"\n📊 Timing Statistics for {model_name}:")
    print(f"  Average inference time: {avg_time:.3f} ± {std_time:.3f} seconds")
    print(f"  Min inference time: {min_time:.3f} seconds")
    print(f"  Max inference time: {max_time:.3f} seconds")
    
    return {
        'model_name': model_name,
        'num_images': num_images,
        'avg_time': avg_time,
        'std_time': std_time,
        'min_time': min_time,
        'max_time': max_time,
        'all_times': inference_times
    } 

def evaluate_single_image(model, image, true_mask=None, model_name=None):
    """
    Evaluates a model on a single image and returns the results.
    
    Args:
        model: The model to evaluate (TFLite or Keras)
        image: Input image as numpy array in [0,1] range
        true_mask: Optional ground truth mask
        model_name: Name of the model for logging
        
    Returns:
        dict: Dictionary containing:
            - image: Original input image
            - pred_mask: Predicted segmentation mask
            - true_mask: Ground truth mask (if provided)
            - inference_time: Time taken for inference
            - true_classes: Ground truth classes (if true_mask provided)
    """
    results = {}
    results['image'] = image
    
    # Get ground truth classes if mask provided
    if true_mask is not None:
        true_classes = np.unique(true_mask)
        results['true_mask'] = true_mask
        results['true_classes'] = true_classes
        print("\n📊 Ground truth classes:")
        for cls in true_classes:
            print(f"  Class {cls}")
    else:
        true_classes = None
        
    # Run inference and measure time
    pred_mask, inference_time = run_inference_on_image(model, image, model_name, true_classes)
    
    results['pred_mask'] = pred_mask
    results['inference_time'] = inference_time
    
    print(f"⏱️  Inference time: {inference_time:.3f} seconds")
    
    return results

def run_inference_on_arbitrary_image(model, image_path, model_name=None, target_size=(512, 512)):
    """
    Runs inference on any arbitrary image file.
    
    Args:
        model: The model to evaluate (TFLite or Keras)
        image_path: Path to the image file
        model_name: Name of the model for logging
        target_size: Tuple of (height, width) for resizing
        
    Returns:
        dict: Dictionary containing:
            - image: Original input image
            - pred_mask: Predicted segmentation mask
            - inference_time: Time taken for inference
    """
    # Load and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize image
    img = cv2.resize(img, target_size)
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Run inference
    return evaluate_single_image(model, img, model_name=model_name) 