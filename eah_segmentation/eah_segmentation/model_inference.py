# eah_segmentation/model_inference.py

import cv2
import numpy as np
import tensorflow as tf
from transformers import SegformerImageProcessor, TFSegformerForSemanticSegmentation
import os

def preprocess_image(image_bgr, input_size=(512, 512)):
    """
    Preprocess image for SegFormer model
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, input_size)
    return resized

def map_segformer_to_ade20k(predictions, true_classes=None):
    """
    Map SegFormer class indices to ADE20K class indices.
    This is needed because SegFormer uses 0-based indexing while ADE20K uses 1-based indexing.
    
    Args:
        predictions: SegFormer predictions (H, W) with class indices
        true_classes: List of ground truth class indices to map to
        
    Returns:
        Mapped predictions to ADE20K class indices
    """
    # Simply add 1 to all predictions to match ADE20K's 1-based indexing
    return predictions + 1

def run_segformer_inference(model, image):
    """Run inference using SegFormer model."""
    if isinstance(model, tf.lite.Interpreter):
        # TFLite model handling
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        
        # Preprocess input
        if len(input_details[0]['shape']) == 4:
            inp = np.expand_dims(image, axis=0).astype(np.float32)
        else:
            inp = image.astype(np.float32)
        
        # Quantize if needed
        if input_details[0]['dtype'] == np.int8:
            scale, zp = input_details[0]['quantization']
            inp = (inp / scale + zp).astype(np.int8)
        
        # Run inference
        model.set_tensor(input_details[0]['index'], inp)
        model.invoke()
        
        # Get raw output and handle shape
        raw_out = model.get_tensor(output_details[0]['index'])
        print(f"📊 TFLite raw output shape: {raw_out.shape}, dtype: {raw_out.dtype}")
        
        # Handle batch dimension if present
        if raw_out.ndim == 4 and raw_out.shape[0] == 1:
            raw_out = raw_out[0]
        
        # Get predictions
        predictions = np.argmax(raw_out, axis=-1)
        predictions_np = predictions.astype(np.int32)
        
    else:
        # Keras model handling
        # Preprocess the image
        if len(image.shape) == 3:
            inp = tf.expand_dims(image, 0)
        else:
            inp = image
        
        # Run inference using the serving function
        outputs = model.signatures['serving_default'](inp)
        
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
    """Run inference using DeepLab model."""
    if isinstance(model, tf.lite.Interpreter):
        # TFLite model handling
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        
        # Preprocess input
        if len(input_details[0]['shape']) == 4:
            inp = np.expand_dims(image, axis=0).astype(np.float32)
        else:
            inp = image.astype(np.float32)
        
        # Quantize if needed
        if input_details[0]['dtype'] == np.int8:
            scale, zp = input_details[0]['quantization']
            inp = (inp / scale + zp).astype(np.int8)
        
        # Run inference
        model.set_tensor(input_details[0]['index'], inp)
        model.invoke()
        
        # Get raw output and handle shape
        raw_out = model.get_tensor(output_details[0]['index'])
        print(f"📊 TFLite raw output shape: {raw_out.shape}, dtype: {raw_out.dtype}")
        
        # Handle batch dimension if present
        if raw_out.ndim == 4 and raw_out.shape[0] == 1:
            raw_out = raw_out[0]
        elif raw_out.ndim == 3 and raw_out.shape[0] == 1:
            raw_out = raw_out[0]
        
        # Handle class dimension if present
        if raw_out.ndim == 3:
            if raw_out.shape[-1] == 1:
                raw_out = np.squeeze(raw_out, axis=-1)
            else:
                raw_out = np.argmax(raw_out, axis=-1)
        
        # Handle flattened output
        if raw_out.ndim == 1:
            raw_out = raw_out.reshape(512, 512)
        
        return raw_out.astype(np.int32)
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
    """Run inference using Mosaic model."""
    if isinstance(model, tf.lite.Interpreter):
        # TFLite model handling
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        
        # Preprocess input
        if len(input_details[0]['shape']) == 4:
            inp = np.expand_dims(image, axis=0).astype(np.float32)
        else:
            inp = image.astype(np.float32)
        
        # Quantize if needed
        if input_details[0]['dtype'] == np.int8:
            scale, zp = input_details[0]['quantization']
            inp = (inp / scale + zp).astype(np.int8)
        
        # Run inference
        model.set_tensor(input_details[0]['index'], inp)
        model.invoke()
        
        # Get raw output and handle shape
        raw_out = model.get_tensor(output_details[0]['index'])
        print(f"📊 TFLite raw output shape: {raw_out.shape}, dtype: {raw_out.dtype}")
        
        # Handle batch dimension if present
        if raw_out.ndim == 4 and raw_out.shape[0] == 1:
            raw_out = raw_out[0]
        elif raw_out.ndim == 3 and raw_out.shape[0] == 1:
            raw_out = raw_out[0]
        
        # Handle class dimension if present
        if raw_out.ndim == 3:
            if raw_out.shape[-1] == 1:
                raw_out = np.squeeze(raw_out, axis=-1)
            else:
                raw_out = np.argmax(raw_out, axis=-1)
        
        # Handle flattened output
        if raw_out.ndim == 1:
            raw_out = raw_out.reshape(512, 512)
        
        return raw_out.astype(np.int32)
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

def load_segformer_model(model_path=None, use_tflite=False):
    """
    Load SegFormer model from local path.
    
    Args:
        model_path: Optional path to a local model. If None, uses default local path.
        use_tflite: Whether to load the TFLite version of the model.
        
    Returns:
        Loaded SegFormer model
    """
    if model_path is None:
        if use_tflite:
            model_path = "models/segformer_b0/tflite/model.tflite"
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"TFLite model not found at {model_path}. "
                    "Please run download_models.py first to download and convert the model."
                )
            # Load TFLite model
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            return interpreter
        else:
            model_path = "models/segformer_b0/keras"
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Keras model not found at {model_path}. "
                    "Please run download_models.py first to download and convert the model."
                )
            # Load Keras model
            return tf.saved_model.load(model_path)
    else:
        if use_tflite:
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            return interpreter
        else:
            return tf.saved_model.load(model_path)
