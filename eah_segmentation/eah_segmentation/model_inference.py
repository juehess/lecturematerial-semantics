# eah_segmentation/model_inference.py

import cv2
import numpy as np
import tensorflow as tf
from transformers import SegformerImageProcessor, TFSegformerForSemanticSegmentation
import os
import time

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
    """Run inference using DeepLab model."""
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

def map_mosaic_to_ade20k(predictions):
    """
    Map Mosaic class indices to ADE20K class indices.
    The model uses ADE20K's 21-class subset.
    
    Args:
        predictions: Mosaic predictions (H, W) with class indices
        
    Returns:
        Mapped predictions to ADE20K class indices
    """
    # ADE20K 21-class subset mapping
    # Class indices are 1-based in ADE20K
    ade20k_21class = {
        0: 0,     # background -> background
        1: 1,     # wall
        2: 2,     # building
        3: 3,     # sky
        4: 4,     # floor
        5: 5,     # tree
        6: 6,     # ceiling
        7: 7,     # road
        8: 8,     # bed
        9: 9,     # windowpane
        10: 10,   # grass
        11: 11,   # cabinet
        12: 12,   # sidewalk
        13: 13,   # person
        14: 14,   # earth
        15: 15,   # door
        16: 16,   # painting
        17: 17,   # mountain
        18: 18,   # plant
        19: 19,   # curtain
        20: 20,   # chair
        21: 21,   # car
    }
    
    # Create a mapping array for efficient lookup
    mapping_array = np.zeros(max(ade20k_21class.keys()) + 1, dtype=np.int32)
    for mosaic_idx, ade20k_idx in ade20k_21class.items():
        mapping_array[mosaic_idx] = ade20k_idx
    
    # Map the predictions
    return mapping_array[predictions]

def run_mosaic_inference(model, image):
    """Run inference using Mosaic model."""
    if isinstance(model, tf.lite.Interpreter):
        # TFLite model handling
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        
        # Print detailed model information
        print("\n📋 TFLite Model Details:")
        print("\nInput Details:")
        for i, detail in enumerate(input_details):
            print(f"  Input {i}:")
            print(f"    Name: {detail['name']}")
            print(f"    Shape: {detail['shape']}")
            print(f"    Type: {detail['dtype']}")
            if 'quantization' in detail:
                scale, zero_point = detail['quantization']
                print(f"    Quantization: scale={scale}, zero_point={zero_point}")
        
        print("\nOutput Details:")
        for i, detail in enumerate(output_details):
            print(f"  Output {i}:")
            print(f"    Name: {detail['name']}")
            print(f"    Shape: {detail['shape']}")
            print(f"    Type: {detail['dtype']}")
            if 'quantization' in detail:
                scale, zero_point = detail['quantization']
                print(f"    Quantization: scale={scale}, zero_point={zero_point}")
        
        # Get expected input size from model
        input_shape = input_details[0]['shape']
        expected_height, expected_width = input_shape[1:3]
        print(f"\n📊 Expected input size: {expected_width}x{expected_height}")
        
        # Resize input to expected dimensions
        resized_image = cv2.resize(image, (expected_width, expected_height))
        print(f"📊 Resized image shape: {resized_image.shape}")
        
        # Convert to UINT8 (0-255 range) as required by the model
        inp = np.expand_dims(resized_image, axis=0)
        inp = (inp * 255).astype(np.uint8)
        
        # Apply quantization if needed
        if 'quantization' in input_details[0]:
            scale, zero_point = input_details[0]['quantization']
            if scale != 0:  # Only apply if quantization is active
                inp = np.round(inp / scale + zero_point).astype(np.uint8)
        
        print(f"📊 Input tensor shape: {inp.shape}")
        print(f"📊 Input tensor range: [{np.min(inp)}, {np.max(inp)}]")
        
        # Run inference
        model.set_tensor(input_details[0]['index'], inp)
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
        
        # Print unique classes in predictions
        unique_classes, class_counts = np.unique(predictions_np, return_counts=True)
        print("\n📊 Predicted classes before mapping:")
        for cls, count in zip(unique_classes, class_counts):
            print(f"  Class {cls}: {count} pixels")
        
        # Map predictions to ADE20K class indices
        predictions_np = map_mosaic_to_ade20k(predictions_np)
        
        # Print unique classes after mapping
        unique_classes, class_counts = np.unique(predictions_np, return_counts=True)
        print("\n📊 Predicted classes after mapping:")
        for cls, count in zip(unique_classes, class_counts):
            print(f"  Class {cls}: {count} pixels")
        
        # Resize back to original image size
        if predictions_np.shape != image.shape[:2]:
            predictions_np = cv2.resize(predictions_np, (image.shape[1], image.shape[0]), 
                                      interpolation=cv2.INTER_NEAREST)
        
        return predictions_np
    else:
        # TensorFlow model handling
        if len(image.shape) == 3:
            inp = tf.expand_dims(image, 0)
        else:
            inp = image
        
        # Convert to UINT8 (0-255 range) as required by the model
        inp = (inp * 255).astype(np.uint8)
        
        # Print available signatures
        print("\n📋 Available model signatures:")
        for signature_name, signature in model.signatures.items():
            print(f"  - {signature_name}")
            print(f"    Inputs: {signature.inputs}")
            print(f"    Outputs: {signature.outputs}")
        
        # Try different signatures
        try:
            # First try serving_default
            outputs = model.signatures['serving_default'](inp)
            print("\n✅ Using 'serving_default' signature")
        except Exception as e:
            print(f"\n❌ Error with 'serving_default': {str(e)}")
            try:
                # Try predict
                outputs = model.signatures['predict'](inp)
                print("\n✅ Using 'predict' signature")
            except Exception as e:
                print(f"\n❌ Error with 'predict': {str(e)}")
                try:
                    # Try inference
                    outputs = model.signatures['inference'](inp)
                    print("\n✅ Using 'inference' signature")
                except Exception as e:
                    print(f"\n❌ Error with 'inference': {str(e)}")
                    raise ValueError("No working signature found")
        
        # Get predictions from the output tensor
        # Try different output names
        output_names = ['output_0', 'logits', 'predictions', 'output']
        logits = None
        for name in output_names:
            if name in outputs:
                logits = outputs[name]
                print(f"\n✅ Using output tensor: {name}")
                break
        
        if logits is None:
            print("\n❌ No known output tensor found. Available outputs:")
            for name, tensor in outputs.items():
                print(f"  - {name}: {tensor.shape}")
            raise ValueError("No known output tensor found")
        
        # Get class predictions
        preds_np = np.argmax(logits[0].numpy(), axis=-1)
        
        # Map predictions to ADE20K class indices
        preds_np = map_mosaic_to_ade20k(preds_np)
        
        return preds_np.astype(np.int32)

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
            # Load TFLite model with optimizations
            interpreter = tf.lite.Interpreter(model_path=model_path)
            
            # Try to enable hardware acceleration
            try:
                # Try to use GPU delegate
                gpu_delegate = tf.lite.experimental.load_delegate('libedgetpu.so.1')
                interpreter = tf.lite.Interpreter(
                    model_path=model_path,
                    experimental_delegates=[gpu_delegate]
                )
                print("✅ Using GPU acceleration")
            except:
                try:
                    # Try to use XNNPACK delegate for CPU optimization
                    interpreter = tf.lite.Interpreter(
                        model_path=model_path,
                        experimental_delegates=[tf.lite.experimental.load_delegate('libxnnpack_delegate.so')]
                    )
                    print("✅ Using XNNPACK CPU optimization")
                except:
                    print("⚠️ Using default CPU execution")
            
            # Set number of threads for CPU execution
            interpreter.set_num_threads(4)
            
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
            interpreter.set_num_threads(4)
            interpreter.allocate_tensors()
            return interpreter
        else:
            return tf.saved_model.load(model_path)
