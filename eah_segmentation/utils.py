"""Utility functions for model loading and other common operations."""

import os
import time
import json
from pathlib import Path
import tensorflow as tf

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