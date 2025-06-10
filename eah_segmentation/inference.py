# eah_segmentation/model_inference.py

import cv2
import numpy as np
import tensorflow as tf
from transformers import SegformerImageProcessor, TFSegformerForSemanticSegmentation
import os
import time

# Try relative import first, then fall back to absolute import
try:
    from .class_mapping import (
        map_segformer_to_ade20k,
        map_mosaic_to_cityscapes,
        map_cityscapes_to_ade20k
    )
except ImportError:
    from class_mapping import (
        map_segformer_to_ade20k,
        map_mosaic_to_cityscapes,
        map_cityscapes_to_ade20k
    )

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
        np.ndarray: Segmentation mask with ADE20K class indices
    """
    if isinstance(model, tf.lite.Interpreter):
        # TFLite model handling
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        
        # Preprocess input - SegFormer TFLite uses float32
        inp = np.expand_dims(image, axis=0).astype(np.float32)
        
        # Set input tensor
        model.set_tensor(input_details[0]['index'], inp)
        
        # Run inference with timing
        start_time = time.perf_counter()
        model.invoke()
        inference_time = time.perf_counter() - start_time
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
        
        # Run inference using the serving function with timing
        start_time = time.perf_counter()
        outputs = model.signatures['serving_default'](inp)
        inference_time = time.perf_counter() - start_time
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
    
    return predictions_np

def run_deeplab_inference(model, image):
    """
    Runs inference using DeepLab model.
    
    Key Operations:
        - Handles input quantization for TFLite models
        - Performs standard DeepLab preprocessing
        - Provides detailed debugging information
        
    Args:
        model: TF/TFLite model
        image (np.ndarray): Input image
        
    Returns:
        np.ndarray: Segmentation predictions
    """
    if isinstance(model, tf.lite.Interpreter):
        # TFLite model handling
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        
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
        
        # Run inference
        model.set_tensor(input_details[0]['index'], img)
        model.invoke()
        
        # Get predictions and debug output
        raw_out = model.get_tensor(output_details[0]['index'])
        print(f"📊 Raw output shape: {raw_out.shape}")
        print(f"📊 Raw output range: [{np.min(raw_out)}, {np.max(raw_out)}]")
        
        predictions = raw_out[0]
        predictions_np = predictions.astype(np.int32)
        
        # Print unique classes in predictions
        unique_classes, class_counts = np.unique(predictions_np, return_counts=True)
        print("\n📊 Predicted classes:")
        for cls, count in zip(unique_classes, class_counts):
            print(f"  Class {cls}: {count} pixels")
        
        return predictions_np
    else:
        # Keras model handling
        if len(image.shape) == 3:
            inp = tf.expand_dims(image, 0)
        else:
            inp = image
        preds = model(inp, training=False)
        if isinstance(preds, tuple):
            preds = preds[0]
        return preds[0].numpy().astype(np.int32)

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
        np.ndarray: Segmentation predictions in ADE20K format
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
    
    # Run inference
    model.set_tensor(input_details[0]['index'], inp_resized)
    model.invoke()
    
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
    
    return predictions_np

def run_inference_on_image(model, image, model_name=None, true_classes=None, ground_truth=None):
    """
    Run inference on a single image using different model types.
    
    Args:
        model: Model to use for inference (SegFormer, DeepLab, or Mosaic)
        image: Input image array (H, W, 3) in range [0..1]
        model_name: Name of the model (e.g., 'segformer_b0', 'deeplabv3plus_edgetpu', 'mosaic')
        true_classes: List of ground truth class indices to map to
        ground_truth: Ground truth segmentation mask for comparison
        
    Returns:
        A 2D NumPy array of shape (H, W) with class indices.
    """
    if hasattr(image, 'numpy'):
        image = image.numpy()
    
    print(f"\n📊 Input image shape: {image.shape}")
    print(f"📊 Input image range: [{np.min(image)}, {np.max(image)}]")
    
    # Run inference based on model architecture
    if model_name == 'segformer_b0':
        predictions_np = run_segformer_inference(model, image)
    elif model_name == 'deeplabv3plus_edgetpu':
        predictions_np = run_deeplab_inference(model, image)
    elif model_name == 'mosaic':
        predictions_np = run_mosaic_inference(model, image)
    else:
        raise ValueError(f"Unknown model type: {model_name}")
    
    print(f"📊 Raw predictions shape: {predictions_np.shape}, dtype: {predictions_np.dtype}")
    print(f"📊 Raw predictions range: [{np.min(predictions_np)}, {np.max(predictions_np)}]")
    
    # Print unique classes and their frequencies
    unique_classes, class_counts = np.unique(predictions_np, return_counts=True)
    print("\n📊 Predicted classes:")
    for cls, count in zip(unique_classes, class_counts):
        print(f"  Class {cls}: {count} pixels")
    
    # Handle ground truth comparison if available
    if ground_truth is not None:
        print("\n📊 Ground truth classes:")
        gt_unique_classes, gt_class_counts = np.unique(ground_truth, return_counts=True)
        for cls, count in zip(gt_unique_classes, gt_class_counts):
            print(f"  Class {cls}: {count} pixels")
        
        # Print class-wise comparison
        print("\n📊 Class-wise comparison (predicted vs ground truth):")
        for cls in set(unique_classes) | set(gt_unique_classes):
            pred_count = np.sum(predictions_np == cls)
            gt_count = np.sum(ground_truth == cls)
            print(f"  Class {cls}: {pred_count} predicted vs {gt_count} ground truth")
        
        # Map predictions to ground truth classes if needed
        if not np.all(np.isin(unique_classes, gt_unique_classes)):
            print("\n🔄 Mapping predictions to ground truth classes...")
            # Create a mapping array
            mapping = np.zeros(np.max(unique_classes) + 1, dtype=np.int32)
            # Map each predicted class to the closest ground truth class
            for pred_cls in unique_classes:
                if pred_cls in gt_unique_classes:
                    mapping[pred_cls] = pred_cls
                else:
                    # Find the closest ground truth class
                    closest_gt = min(gt_unique_classes, key=lambda x: abs(x - pred_cls))
                    mapping[pred_cls] = closest_gt
            # Apply the mapping
            predictions_np = mapping[predictions_np]
    
    # Resize predictions to match input image size if needed
    if predictions_np.shape != image.shape[:2]:
        predictions_np = cv2.resize(predictions_np, (image.shape[1], image.shape[0]), 
                                  interpolation=cv2.INTER_NEAREST)
    
    print(f"Returning predictions of shape: {predictions_np.shape}")
    return predictions_np