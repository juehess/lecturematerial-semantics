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
    ram_usage = []  # Add RAM usage tracking
    
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
        
        # Get ground truth classes and counts if available
        has_ground_truth = true_mask is not None
        if has_ground_truth:
            true_mask_np = true_mask.numpy()[0]
            true_classes, true_counts = np.unique(true_mask_np, return_counts=True)
        
        # Run inference and measure time
        pred_mask, inference_time, ram_delta = run_inference_on_image(
            model, 
            image.numpy()[0], 
            model_name, 
            true_classes if has_ground_truth else None
        )
        inference_times.append(inference_time)
        ram_usage.append(ram_delta)
        
        # Get predicted classes and counts
        pred_classes, pred_counts = np.unique(pred_mask, return_counts=True)
        
        # Get class names from ADE20K mapping
        try:
            from eah_segmentation.ade20k_utils import ADE20K_CONFIG
            class_names = ADE20K_CONFIG['class_names']
            # Add background class (0) if not present
            if 0 not in class_names:
                class_names = ['background'] + class_names
        except ImportError:
            class_names = {i: f"Class_{i}" for i in range(151)}  # Fallback if mapping not available
        
        # Print ground truth table if available
        if has_ground_truth:
            print("\n📊 Ground Truth Classes:")
            print("Class ID | Class Name | Pixel Count")
            print("-" * 40)
            for cls, count in zip(true_classes, true_counts):
                class_name = class_names[int(cls)] if isinstance(class_names, list) else class_names.get(int(cls), f"Unknown_{cls}")
                print(f"{cls:8d} | {class_name:12s} | {count:10d}")
        
        # Print predictions table
        print("\n📊 Predicted Classes:")
        print("Class ID | Class Name | Pixel Count")
        print("-" * 40)
        for cls, count in zip(pred_classes, pred_counts):
            class_name = class_names[int(cls)] if isinstance(class_names, list) else class_names.get(int(cls), f"Unknown_{cls}")
            print(f"{cls:8d} | {class_name:12s} | {count:10d}")
        
        # Print comparison if ground truth is available
        if has_ground_truth:
            print("\n📊 Class Comparison:")
            print("Class ID | Class Name | GT Pixels | Pred Pixels | Match")
            print("-" * 65)
            all_classes = set(true_classes) | set(pred_classes)
            for cls in sorted(all_classes):
                gt_count = true_counts[true_classes == cls][0] if cls in true_classes else 0
                pred_count = pred_counts[pred_classes == cls][0] if cls in pred_classes else 0
                class_name = class_names[int(cls)] if isinstance(class_names, list) else class_names.get(int(cls), f"Unknown_{cls}")
                match = "✓" if gt_count > 0 and pred_count > 0 else "✗"
                print(f"{cls:8d} | {class_name:12s} | {gt_count:9d} | {pred_count:10d} | {match}")
        
        print(f"\n⏱️  Inference time: {inference_time:.3f} seconds")
        print(f"💾 RAM usage: {ram_delta} kB")
        
        # Save visualization if ground truth is available
        if has_ground_truth:
            save_prediction(
                image.numpy()[0],
                true_mask_np,
                pred_mask,
                output_dir,
                i,
                model_type=model_type
            )
        else:
            # Save only prediction if no ground truth
            save_prediction(
                image.numpy()[0],
                None,
                pred_mask,
                output_dir,
                i,
                model_type=model_type
            )
        
        print(f"✅ Processed image {i+1}/{num_images}")
    
    # Calculate and print timing statistics
    avg_time = float(np.mean(inference_times))
    std_time = float(np.std(inference_times))
    min_time = float(np.min(inference_times))
    max_time = float(np.max(inference_times))
    
    # Calculate RAM statistics
    avg_ram = float(np.mean(ram_usage))
    std_ram = float(np.std(ram_usage))
    min_ram = float(np.min(ram_usage))
    max_ram = float(np.max(ram_usage))
    
    # Convert lists to Python native types
    inference_times = [float(t) for t in inference_times]
    ram_usage = [float(r) for r in ram_usage]
    
    print(f"\n📊 Timing Statistics for {model_name}:")
    print(f"  Average inference time: {avg_time:.3f} ± {std_time:.3f} seconds")
    print(f"  Min inference time: {min_time:.3f} seconds")
    print(f"  Max inference time: {max_time:.3f} seconds")
    print(f"\n💾 RAM Usage Statistics:")
    print(f"  Average RAM usage: {avg_ram:.1f} ± {std_ram:.1f} kB")
    print(f"  Min RAM usage: {min_ram:.1f} kB")
    print(f"  Max RAM usage: {max_ram:.1f} kB")
    
    return {
        'model_name': model_name,
        'num_images': num_images,
        'avg_time': avg_time,
        'std_time': std_time,
        'min_time': min_time,
        'max_time': max_time,
        'all_times': inference_times,
        'avg_ram': avg_ram,
        'std_ram': std_ram,
        'min_ram': min_ram,
        'max_ram': max_ram,
        'all_ram': ram_usage
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
    pred_mask, inference_time, ram_delta = run_inference_on_image(model, image, model_name, true_classes)
    
    results['pred_mask'] = pred_mask
    results['inference_time'] = inference_time
    results['ram_usage'] = ram_delta
    
    print(f"⏱️  Inference time: {inference_time:.3f} seconds")
    print(f"💾 RAM usage: {ram_delta} kB")
    
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
    results = evaluate_single_image(model, img, model_name=model_name)
    return results 