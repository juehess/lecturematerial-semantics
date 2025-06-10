"""Model evaluation utilities."""

import os
import tensorflow as tf
import numpy as np
from pathlib import Path
import argparse
import time
import json
from datetime import datetime

# Try relative imports first, fall back to absolute imports
try:
    from .ade20k_utils import load_ade20k_dataset, save_prediction
    from .inference import run_inference_on_image
except ImportError:
    from ade20k_utils import load_ade20k_dataset, save_prediction
    from inference import run_inference_on_image

def load_model(model_name, model_type='keras', device='cpu'):
    """
    Load a model for evaluation.
    
    Args:
        model_name (str): Name of the model to load
        model_type (str): Type of model ('tflite' or 'keras')
        device (str): Device to run on ('cpu' or 'coral')
        
    Returns:
        tuple: (model, load_time)
            - model: The loaded model (TFLite interpreter or Keras model)
            - load_time: Time taken to load the model in seconds
    """
    print(f"\n🚀 Loading {model_name} ({model_type}) for {device}...")
    
    # Find model directory
    model_dir = Path(__file__).parent.parent / 'models' / model_name
    if not model_dir.exists():
        raise ValueError(f"Model directory not found: {model_dir}")
    
    # Measure model loading time
    load_start_time = time.perf_counter()
    
    if model_type == 'tflite':
        # Load TFLite model
        tflite_path = model_dir / 'tflite' / '1.tflite'
        if not tflite_path.exists():
            raise ValueError(f"TFLite model not found: {tflite_path}")
            
        print(f"📥 Loading TFLite model from {tflite_path}")
        if device == 'coral':
            try:
                interpreter = tf.lite.Interpreter(
                    model_path=str(tflite_path),
                    experimental_delegates=[tf.lite.load_delegate('libedgetpu.so.1')]
                )
                print("✅ Successfully loaded model on Coral TPU")
            except Exception as e:
                print(f"❌ Failed to load model on Coral TPU: {str(e)}")
                print("⚠️ Falling back to CPU")
                interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        else:
            interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        model = interpreter
    else:
        # Load Keras/SavedModel
        keras_path = model_dir / 'keras'
        if not keras_path.exists():
            raise ValueError(f"Keras model not found: {keras_path}")
            
        print(f"📥 Loading Keras model from {keras_path}")
        model = tf.saved_model.load(str(keras_path))
        if model is None:
            raise ValueError(f"Failed to load model: {model_name}")
    
    load_time = time.perf_counter() - load_start_time
    print(f"⏱️  Model loading time: {load_time:.3f} seconds")
    
    return model, load_time

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
        
        # Measure inference time
        start_time = time.perf_counter()
        pred_mask = run_inference_on_image(model, image.numpy()[0], model_name, true_classes)
        end_time = time.perf_counter()
        inference_time = end_time - start_time
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

def save_timing_results(results, output_dir):
    """
    Saves model evaluation timing results to JSON.
    
    Args:
        results (dict): Dictionary containing timing statistics
        output_dir (str): Directory to save results
    """
    # Save results in the same directory as predictions
    output_file = Path(output_dir) / 'timing_results.json'
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Timing results saved to {output_file}")

def main():
    """
    Main entry point for model testing.
    
    Key Features:
        - Command-line interface for test configuration
        - Support for multiple models and formats
        - Hardware acceleration selection
        - Comprehensive result logging
        
    Command-line Arguments:
        --models: List of models to test
        --model_type: Model format (tflite/keras)
        --num_images: Number of test images
        --device: Hardware device (cpu/coral)
        --output_dir: Results directory
        --data_dir: Dataset directory
    """
    parser = argparse.ArgumentParser(description='Test segmentation models on ADE20K dataset')
    parser.add_argument('--models', type=str, nargs='+', required=True,
                      help='List of model names to test')
    parser.add_argument('--model_type', type=str, choices=['tflite', 'keras'], default='keras',
                      help='Type of model to use (tflite or keras)')
    parser.add_argument('--num_images', type=int, default=1,
                      help='Number of images to test on (default: 1)')
    parser.add_argument('--output_dir', type=str, default='results',
                      help='Directory to save results')
    parser.add_argument('--data_dir', type=str, default='datasets',
                      help='Directory containing ADE20K dataset')
    parser.add_argument('--image_index', type=int, default=0,
                      help='Index of the image to test (default: 0)')
    parser.add_argument('--device', type=str, choices=['cpu', 'coral'], default='cpu',
                      help='Device to run inference on (cpu or coral)')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load ADE20K dataset
    print(f"📥 Loading ADE20K dataset from {args.data_dir}")
    dataset = load_ade20k_dataset(args.data_dir)
    
    # Skip to the desired image index
    dataset = dataset.skip(args.image_index)
    
    # Store timing results for all models
    all_results = []
    
    # Test each model
    for model_name in args.models:
        # Create model-specific output directory
        model_output_dir = output_dir / f"{model_name}_{args.model_type}_{args.device}"
        model_output_dir.mkdir(exist_ok=True)
        
        try:
            # Load model
            model, load_time = load_model(model_name, args.model_type, args.device)
            
            # Evaluate model
            results = evaluate_model(model, dataset, model_output_dir, args.num_images, model_name)
            
            # Add model loading time to results
            results['load_time'] = load_time
            all_results.append(results)
            
            # Save individual model results
            save_timing_results(results, model_output_dir)
            
        except Exception as e:
            print(f"❌ Error testing {model_name}: {str(e)}")
            continue
    
    # Save combined results
    if all_results:
        combined_results = {
            'timestamp': datetime.now().isoformat(),
            'models': all_results
        }
        save_timing_results(combined_results, output_dir)

if __name__ == '__main__':
    main() 