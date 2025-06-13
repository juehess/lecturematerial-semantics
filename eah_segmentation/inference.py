# eah_segmentation/model_inference.py

import cv2
import numpy as np
import tensorflow as tf
from transformers import SegformerImageProcessor, TFSegformerForSemanticSegmentation
import os
import time
from typing import Tuple, Union
import psutil

try:
    from class_mapping import (
        map_segformer_to_ade20k,
        map_mosaic_to_cityscapes,
        map_cityscapes_to_ade20k,
        map_pascal_to_ade20k
    )
    from model_config import MODEL_NAMES
    from visualization import colorize_mask
except ImportError:
    from eah_segmentation.class_mapping import (
        map_segformer_to_ade20k,
        map_mosaic_to_cityscapes,
        map_cityscapes_to_ade20k,
        map_pascal_to_ade20k
    )
    from eah_segmentation.model_config import MODEL_NAMES
    from visualization import colorize_mask

def run_segformer_inference(model, image):
    """
    Runs inference using SegFormer model (TFLite or Keras).
    
    Key Operations:
        - Handles both TFLite and Keras model formats
        - Preprocesses input appropriately for each format
        - Times inference execution
        - Post-processes predictions to match ADE20K format
    
    Args:
        model: Either tf.lite.Interpreter or keras.Model
        image (np.ndarray): Preprocessed input image
        
    Returns:
        tuple: (predictions, inference_time, ram_delta_kb)
            - predictions (np.ndarray): Segmentation mask with ADE20K class indices
            - inference_time (float): Time taken for inference in seconds
            - ram_delta_kb (float): RAM delta in kilobytes
    """
    if isinstance(model, tf.lite.Interpreter):
        # TFLite model handling
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        
        # Preprocess input - SegFormer TFLite uses float32
        inp = np.expand_dims(image, axis=0).astype(np.float32)
        
        # Set input tensor
        model.set_tensor(input_details[0]['index'], inp)
        
        # Measure RAM before inference
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss
        
        # Run inference with timing
        start_time = time.perf_counter()
        model.invoke()
        inference_time = time.perf_counter() - start_time
        mem_after = process.memory_info().rss
        ram_delta_kb = (mem_after - mem_before) // 1024
        print(f"⏱️ TFLite inference time: {inference_time*1000:.2f}ms")
        
        # Get predictions directly
        raw_out = model.get_tensor(output_details[0]['index'])
        predictions = np.argmax(raw_out[0], axis=-1)
        predictions_np = predictions.astype(np.int32)
        
    else:
        # Keras model handling
        # Preprocess the image
        if len(image.shape) == 3:
            inp = tf.expand_dims(image, 0)
        else:
            inp = image
        
        # Measure RAM before inference
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss
        
        # Run inference using the serving function with timing
        start_time = time.perf_counter()
        outputs = model.signatures['serving_default'](inp)
        inference_time = time.perf_counter() - start_time
        mem_after = process.memory_info().rss
        ram_delta_kb = (mem_after - mem_before) // 1024
        print(f"⏱️ Keras inference time: {inference_time*1000:.2f}ms")
        
        # Get predictions from the output tensor
        logits = outputs['output_0']  # The output tensor name from the SavedModel
        
        predictions = tf.argmax(logits, axis=-1)
        predictions_np = predictions[0].numpy().astype(np.int32)
    
    # Handle different output shapes
    if len(predictions_np.shape) == 1:
        # Reshape to 512x512
        predictions_np = predictions_np.reshape(512, 512)
    
    # Map predictions to ADE20K class indices
    predictions_np = map_segformer_to_ade20k(predictions_np)
    
    # Resize to match input image size if needed
    if predictions_np.shape != image.shape[:2]:
        predictions_np = cv2.resize(predictions_np, (image.shape[1], image.shape[0]), 
                                  interpolation=cv2.INTER_NEAREST)
    
    return predictions_np, inference_time, ram_delta_kb

def run_deeplab_inference(model, image):
    """
    Runs inference using DeepLab model.
    
    Key Operations:
        - Handles input quantization for TFLite models:
          * Resize input to model's expected size
          * Convert to [-1, 1] range
          * Quantize to int8 range [-128, 127]
        - Performs standard DeepLab preprocessing
        - Provides detailed debugging information
        
    Args:
        model: TF/TFLite model
        image (np.ndarray): Input image in [0,1] range
        
    Returns:
        tuple: (predictions, inference_time, ram_delta_kb)
            - predictions (np.ndarray): Segmentation predictions
            - inference_time (float): Time taken for inference in seconds
            - ram_delta_kb (float): RAM delta in kilobytes
    """
    if isinstance(model, tf.lite.Interpreter):
        # TFLite model handling
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        
        # Print model details
        print("\n📊 Model Details:")
        print("Input Details:")
        for detail in input_details:
            print(f"  Name: {detail['name']}")
            print(f"  Shape: {detail['shape']}")
            print(f"  Type: {detail['dtype']}")
            print(f"  Quantization: {detail['quantization']}")
        
        print("\nOutput Details:")
        for detail in output_details:
            print(f"  Name: {detail['name']}")
            print(f"  Shape: {detail['shape']}")
            print(f"  Type: {detail['dtype']}")
            print(f"  Quantization: {detail['quantization']}")
        
        # Get expected input size from model
        input_shape = input_details[0]['shape']
        expected_height, expected_width = input_shape[1:3]
        
        # Resize input to expected dimensions
        resized_image = cv2.resize(image, (expected_width, expected_height))
        
        # Debug original input
        print(f"📊 Original input range: [{np.min(resized_image)}, {np.max(resized_image)}]")
        
        # Standard DeepLab preprocessing:
        # 1. Convert to float32
        # 2. Normalize to [-1, 1]
        img = resized_image.astype(np.float32)
        img = img * 2.0 - 1.0  # [0,1] -> [-1,1]
        
        # Debug normalized range
        print(f"📊 Normalized range: [{np.min(img)}, {np.max(img)}]")
        
        # 3. Apply quantization
        scale, zero_point = input_details[0]['quantization']
        img = np.round(img / scale + zero_point)
        img = np.clip(img, -128, 127).astype(np.int8)
        
        # Debug quantized range
        print(f"📊 Quantized range: [{np.min(img)}, {np.max(img)}]")
        
        img = np.expand_dims(img, 0)
        
        # Measure RAM before inference
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss
        
        # Run inference with timing
        model.set_tensor(input_details[0]['index'], img)
        start_time = time.perf_counter()
        model.invoke()
        output = model.get_tensor(output_details[0]['index'])
        inference_time = time.perf_counter() - start_time
        mem_after = process.memory_info().rss
        ram_delta_kb = (mem_after - mem_before) // 1024
        
        print(f"⏱️ TFLite inference time: {inference_time*1000:.2f}ms")
        
        # Process output
        print(f"📊 Raw output shape: {output.shape}")
        print(f"📊 Raw output range: [{np.min(output)}, {np.max(output)}]")
        print(f"📊 Raw output unique values: {np.unique(output)}")
        
        # Get predictions - output already contains class indices
        predictions = output[0].astype(np.int32)
        
        # Print unique classes in predictions
        unique_classes, class_counts = np.unique(predictions, return_counts=True)
        print("\n📊 Predicted classes:")
        for cls, count in zip(unique_classes, class_counts):
            print(f"  Class {cls}: {count} pixels")
        
        # Resize predictions back to original size if needed
        if predictions.shape != image.shape[:2]:
            predictions = cv2.resize(predictions.astype(np.float32), 
                                  (image.shape[1], image.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
        
        return predictions.astype(np.int32), inference_time, ram_delta_kb
    else:
        # Keras model handling
        if len(image.shape) == 3:
            inp = tf.expand_dims(image, 0)
        else:
            inp = image
            
        # Measure RAM before inference
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss
        
        # Run inference with timing
        start_time = time.perf_counter()
        preds = model(inp, training=False)
        inference_time = time.perf_counter() - start_time
        mem_after = process.memory_info().rss
        ram_delta_kb = (mem_after - mem_before) // 1024
        print(f"⏱️ Keras inference time: {inference_time*1000:.2f}ms")
        
        if isinstance(preds, tuple):
            preds = preds[0]
        return preds[0].numpy().astype(np.int32), inference_time, ram_delta_kb

def run_mosaic_inference(model, image):
    """
    Runs inference using Mosaic model (TFLite only).
    
    Key Operations:
        - Handles TFLite model format
        - Maps predictions to Cityscapes classes
        - Maps Cityscapes classes to ADE20K format
        
    Args:
        model (tf.lite.Interpreter): TFLite model interpreter
        image (np.ndarray): Input image in range [0, 1]
        
    Returns:
        tuple: (predictions, inference_time, ram_delta_kb)
            - predictions (np.ndarray): Segmentation predictions in ADE20K format
            - inference_time (float): Time taken for inference in seconds
            - ram_delta_kb (float): RAM delta in kilobytes
    """
    if not isinstance(model, tf.lite.Interpreter):
        raise ValueError("Mosaic model must be a TFLite model")
        
    # TFLite model handling
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    
    # Get expected input size from model
    input_shape = input_details[0]['shape']
    expected_height, expected_width = input_shape[1:3]
    
    # Convert input from float32 [0,1] to uint8 [0,255]
    inp = np.expand_dims(image, axis=0)
    inp = (inp * 255).astype(np.uint8)
    
    # Resize to model's expected size
    inp_resized = np.zeros(input_shape, dtype=np.uint8)
    inp_resized[0] = cv2.resize(inp[0], (expected_width, expected_height), 
                              interpolation=cv2.INTER_LINEAR)
    
    print(f"📊 Input tensor shape: {inp_resized.shape}")
    print(f"📊 Input tensor range: [{np.min(inp_resized)}, {np.max(inp_resized)}]")
    
    # Measure RAM before inference
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss
    
    # Run inference with timing
    model.set_tensor(input_details[0]['index'], inp_resized)
    start_time = time.perf_counter()
    model.invoke()
    inference_time = time.perf_counter() - start_time
    mem_after = process.memory_info().rss
    ram_delta_kb = (mem_after - mem_before) // 1024
    print(f"⏱️ TFLite inference time: {inference_time*1000:.2f}ms")
    
    # Get predictions directly
    raw_out = model.get_tensor(output_details[0]['index'])
    print(f"\n📊 Raw output shape: {raw_out.shape}")
    print(f"📊 Raw output range: [{np.min(raw_out)}, {np.max(raw_out)}]")
    
    # Dequantize output if needed
    if 'quantization' in output_details[0]:
        scale, zero_point = output_details[0]['quantization']
        if scale != 0:  # Only apply if quantization is active
            raw_out = (raw_out.astype(np.float32) - zero_point) * scale
    
    predictions = np.argmax(raw_out[0], axis=-1)
    predictions_np = predictions.astype(np.int32)
    print(f"📊 Predictions shape: {predictions_np.shape}")
    print(f"📊 Predictions range: [{np.min(predictions_np)}, {np.max(predictions_np)}]")
    
    # Print unique classes in predictions (raw Cityscapes classes)
    unique_classes, class_counts = np.unique(predictions_np, return_counts=True)
    print("\n📊 Predicted Cityscapes classes:")
    for cls, count in zip(unique_classes, class_counts):
        print(f"  Class {cls}: {count} pixels")
    
    # Map Cityscapes predictions to ADE20k classes
    predictions_np = map_cityscapes_to_ade20k(predictions_np)
    
    # Print unique classes after mapping to ADE20k
    unique_classes, class_counts = np.unique(predictions_np, return_counts=True)
    print("\n📊 Predicted ADE20k classes:")
    for cls, count in zip(unique_classes, class_counts):
        print(f"  Class {cls}: {count} pixels")
    
    # Resize back to original image size using nearest neighbor to preserve class indices
    if predictions_np.shape != image.shape[:2]:
        predictions_np = cv2.resize(predictions_np, (image.shape[1], image.shape[0]), 
                                  interpolation=cv2.INTER_NEAREST)
    
    return predictions_np, inference_time, ram_delta_kb

def quantize_input(img, scale, zero_point, input_details):
    """
    Quantizes input image based on the model's actual input requirements.
    
    Args:
        img (np.ndarray): Input image in [0,1] range
        scale (float): Quantization scale
        zero_point (int): Quantization zero point
        input_details (dict): Model input details
        
    Returns:
        np.ndarray: Quantized input image
    """
    # Get actual input type after tensor allocation
    input_type = input_details['dtype']
    print(f"\n📊 Model expects input type: {input_type}")
    
    # For uint8, we can directly scale from [0,1] to [0,255]
    if input_type == np.uint8:
        print("🔄 Quantizing to uint8 [0, 255]")
        img = np.round(img * 255).astype(np.uint8)
    # For int8, we need to use the quantization parameters
    elif input_type == np.int8:
        print("🔄 Quantizing to int8 [-128, 127]")
        img = np.round(img / scale + zero_point)
        img = np.clip(img, -128, 127).astype(np.int8)
    
    return img

def run_deeplabv3_cityscapes_inference(model, image):
    """
    Runs inference using DeepLab v3 Cityscapes model.
    
    Key Operations:
        - Handles input preprocessing for Cityscapes model:
          * Resize input to 513x513 (model's native size)
          * Convert directly to uint8 [0,255] (no normalization needed)
          * Maintain aspect ratio with padding if needed
        - Performs inference with timing
        - Maps predictions to ADE20K format
        - Removes padding and resizes back to input size
        
    Args:
        model: TFLite model
        image (np.ndarray): Input image in [0,1] range
        
    Returns:
        tuple: (predictions, inference_time, ram_delta_kb)
            - predictions (np.ndarray): Predictions mapped to ADE20K format
            - inference_time (float): Time taken for inference in seconds
            - ram_delta_kb (float): RAM delta in kilobytes
    """
    #if not isinstance(model, tf.lite.Interpreter):
    #    raise ValueError("DeepLabV3 Cityscapes model must be a TFLite model")
        
    # TFLite model handling
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    
    # Get expected input size from model
    input_shape = input_details[0]['shape']
    expected_height, expected_width = input_shape[1:3]
    
    # Calculate resize dimensions while preserving aspect ratio
    h, w = image.shape[:2]
    scale = min(expected_height/h, expected_width/w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Resize using LANCZOS for better quality
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Create padded image (black padding)
    resized_image = np.zeros((expected_height, expected_width, 3), dtype=np.float32)
    y_offset = (expected_height - new_h) // 2
    x_offset = (expected_width - new_w) // 2
    resized_image[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    # Convert to uint8 [0,255] - this is what the model expects
    img = (resized_image * 255).astype(np.uint8)
    print(f"\n📊 Input uint8 range: [{np.min(img)}, {np.max(img)}]")
    
    # Add batch dimension
    img = np.expand_dims(img, 0)
    
    # Measure RAM before inference
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss
    
    # Run inference with timing
    model.set_tensor(input_details[0]['index'], img)
    start_time = time.perf_counter()
    model.invoke()
    output = model.get_tensor(output_details[0]['index'])
    inference_time = time.perf_counter() - start_time
    mem_after = process.memory_info().rss
    ram_delta_kb = (mem_after - mem_before) // 1024
    
    print(f"⏱️ TFLite inference time: {inference_time*1000:.2f}ms")
    
    # Process output
    print(f"📊 Raw output shape: {output.shape}")
    print(f"📊 Raw output range: [{np.min(output)}, {np.max(output)}]")
    print(f"📊 Raw output unique values: {np.unique(output)}")
    
    # Get predictions - output already contains class indices
    predictions = output[0].astype(np.int32)
    
    # Print unique classes in predictions
    unique_classes, class_counts = np.unique(predictions, return_counts=True)
    print("\n📊 Predicted classes:")
    for cls, count in zip(unique_classes, class_counts):
        print(f"  Class {cls}: {count} pixels")
    
    # Map Cityscapes predictions to ADE20k classes
    predictions = map_cityscapes_to_ade20k(predictions)
    
    # Remove padding from predictions to match input aspect ratio
    predictions = predictions[y_offset:y_offset+new_h, x_offset:x_offset+new_w]
    
    # Resize back to original input size
    predictions = cv2.resize(predictions.astype(np.float32), (w, h), 
                           interpolation=cv2.INTER_NEAREST)
    
    return predictions.astype(np.int32), inference_time, ram_delta_kb

def run_inference_on_image(model, image, model_name=None, true_classes=None, ground_truth=None):
    """Run inference on a single image using the specified model."""
    print(f"\n📊 Input image shape: {image.shape}")
    print(f"📊 Input image range: [{np.min(image)}, {np.max(image)}]")
    
    # Map model name if provided
    if model_name is not None:
        model_name = MODEL_NAMES.get(model_name, model_name)
    
    # Run model-specific inference
    if 'segformer' in model_name:
        predictions, inference_time, ram_delta_kb = run_segformer_inference(model, image)
    elif 'deeplabv3_cityscapes' in model_name:  # Check cityscapes first
        predictions, inference_time, ram_delta_kb = run_deeplabv3_cityscapes_inference(model, image)
    elif 'deeplabv3plus_edgetpu' in model_name:  # Then check general deeplabv3
        predictions, inference_time, ram_delta_kb = run_deeplab_inference(model, image)
    elif 'mosaic' in model_name:
        predictions, inference_time, ram_delta_kb = run_mosaic_inference(model, image)
    else:
        raise ValueError(f"Unknown model type: {model_name}")
        
    return predictions, inference_time, ram_delta_kb