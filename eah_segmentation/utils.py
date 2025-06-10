"""Utility functions for model loading and other common operations."""

import os
import time
import json
from pathlib import Path
import tensorflow as tf
import numpy as np
from datetime import datetime

try:
    from model_config import MODEL_NAMES
except ImportError:
    from eah_segmentation.model_config import MODEL_NAMES

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
    
    # Map model name if it exists in mapping
    model_name = MODEL_NAMES.get(model_name, model_name)
    
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

def save_timing_results(results, output_dir):
    """
    Save timing results to a JSON file.
    
    Args:
        results: Either a single model's results dict or a combined results dict
        output_dir: Directory to save results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle single model results
    if isinstance(results, dict) and 'inference_times' in results:
        inference_times = results['inference_times']
        model_name = results.get('model_name', 'unknown_model')
        
        results = {
            'model_name': model_name,
            'mean_time': float(np.mean(inference_times)),
            'std_time': float(np.std(inference_times)),
            'min_time': float(np.min(inference_times)),
            'max_time': float(np.max(inference_times)),
            'num_runs': len(inference_times),
            'all_times': [float(t) for t in inference_times],
            'load_time': results.get('load_time', None)
        }
        output_file = output_dir / f"{model_name}_timing.json"
    else:
        # Handle combined results
        output_file = output_dir / f"timing_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Timing results saved to {output_file}") 