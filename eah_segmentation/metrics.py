"""Model evaluation and metrics utilities."""

import time
import numpy as np
from pathlib import Path

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
            i
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