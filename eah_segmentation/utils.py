"""Utility functions for model loading and other common operations."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import json
from pathlib import Path
import tensorflow as tf
import numpy as np
from datetime import datetime
from typing import Union, Tuple, Dict, List, Optional
import psutil
import traceback

# Import model config
try:
    from model_config import MODEL_NAMES
except ImportError:
    from eah_segmentation.model_config import MODEL_NAMES

# Import LiteRT if available
USE_LITERT = False
try:
    from ai_edge_litert.interpreter import Interpreter as LiteRTInterpreter
    USE_LITERT = True
except Exception as e:
    print(f"Warning: Failed to import ai_edge_litert:")
    print(f"Exception type: {type(e).__name__}")
    print(f"Exception message: {str(e)}")
    print("Full traceback:")
    traceback.print_exc()
    print("Falling back to tf.lite.Interpreter.")

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
        if device == 'coral':
            # For Coral, look in tflite_edgetpu directory first
            tflite_dir = model_dir / 'tflite_edgetpu'
            if not tflite_dir.exists():
                # Fall back to tflite directory if tflite_edgetpu doesn't exist
                tflite_dir = model_dir / 'tflite'
        else:
            tflite_dir = model_dir / 'tflite'
        
        # Try standardized name first
        tflite_path = tflite_dir / ('1_edgetpu.tflite' if device == 'coral' else '1.tflite')
        
        # If not found, try to find any .tflite file
        if not tflite_path.exists():
            tflite_files = list(tflite_dir.glob('*.tflite'))
            if not tflite_files:
                raise ValueError(f"No TFLite model found in: {tflite_dir}")
            
            # For Coral, prefer files with 'edgetpu' in name
            if device == 'coral':
                edgetpu_files = [f for f in tflite_files if 'edgetpu' in f.name.lower()]
                tflite_path = edgetpu_files[0] if edgetpu_files else tflite_files[0]
            else:
                # For CPU, prefer files without 'edgetpu' in name
                cpu_files = [f for f in tflite_files if 'edgetpu' not in f.name.lower()]
                tflite_path = cpu_files[0] if cpu_files else tflite_files[0]
            
        print(f"📥 Loading TFLite model from {tflite_path}")
        if device == 'coral':
            try:
                from pycoral.utils.edgetpu import make_interpreter
                interpreter = make_interpreter(str(tflite_path))
                interpreter.allocate_tensors()
                print("✅ Successfully loaded model on Coral TPU")
            except Exception as e:
                print(f"❌ Failed to load model on Coral TPU: {str(e)}")
                print("⚠️ Falling back to CPU")
                if USE_LITERT:
                    interpreter = LiteRTInterpreter(model_path=str(tflite_path))
                else:
                    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
                interpreter.allocate_tensors()
        else:
            if USE_LITERT:
                interpreter = LiteRTInterpreter(model_path=str(tflite_path))
            else:
                interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
            interpreter.allocate_tensors()
        model = interpreter
    else:
        # Load Keras model
        keras_dir = model_dir / 'keras'
        if not keras_dir.exists():
            raise ValueError(f"Keras model directory not found: {keras_dir}")
            
        print(f"📥 Loading Keras model from {keras_dir}")
        model = tf.saved_model.load(str(keras_dir))
    
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