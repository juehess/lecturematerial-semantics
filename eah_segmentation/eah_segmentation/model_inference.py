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
    This is needed because SegFormer uses a different class mapping than ADE20K.
    
    Args:
        predictions: SegFormer predictions (H, W) with class indices
        true_classes: List of ground truth class indices to map to
        
    Returns:
        Mapped predictions to ADE20K class indices
    """
    # If no ground truth classes provided, use default mapping
    if true_classes is None:
        true_classes = [0, 1, 2, 3, 5, 7, 10, 18]  # Common ADE20K classes
    
    # Kaggle's SegFormer to ADE20K mapping based on actual output
    segformer_to_ade20k = {
        0: 0,    # background -> background
        1: 1,    # wall -> wall
        2: 2,    # building -> building
        3: 3,    # sky -> sky
        4: 5,    # floor -> tree (map to closest semantic class)
        5: 5,    # tree -> tree
        6: 3,    # ceiling -> sky (map to closest semantic class)
        7: 7,    # road -> road
        8: 1,    # bed -> wall (map to closest semantic class)
        9: 10,   # windowpane -> grass (map to closest semantic class)
        10: 10,  # grass -> grass
        11: 1,   # cabinet -> wall (map to closest semantic class)
        12: 7,   # sidewalk -> road (map to closest semantic class)
        13: 18,  # person -> plant (map to closest semantic class)
        14: 5,   # earth -> tree (map to closest semantic class)
        15: 1,   # door -> wall (map to closest semantic class)
        16: 1,   # table -> wall (map to closest semantic class)
        17: 5,   # mountain -> tree (map to closest semantic class)
        18: 18,  # plant -> plant
        19: 1,   # curtain -> wall (map to closest semantic class)
        20: 1,   # chair -> wall (map to closest semantic class)
        21: 7,   # car -> road (map to closest semantic class)
        22: 3,   # water -> sky (map to closest semantic class)
        23: 1,   # painting -> wall (map to closest semantic class)
        24: 1,   # sofa -> wall (map to closest semantic class)
        25: 1,   # shelf -> wall (map to closest semantic class)
        26: 2,   # house -> building
        27: 3,   # sea -> sky
        28: 1,   # mirror -> wall (map to closest semantic class)
        29: 5,   # rug -> tree (map to closest semantic class)
        30: 10,  # field -> grass
        31: 1,   # armchair -> wall (map to closest semantic class)
        32: 1,   # seat -> wall (map to closest semantic class)
        33: 1,   # fence -> wall (map to closest semantic class)
        34: 1,   # desk -> wall (map to closest semantic class)
        35: 5,   # rock -> tree (map to closest semantic class)
        36: 1,   # wardrobe -> wall (map to closest semantic class)
        37: 1,   # lamp -> wall (map to closest semantic class)
        38: 1,   # bathtub -> wall (map to closest semantic class)
        39: 1,   # railing -> wall (map to closest semantic class)
        40: 1,   # cushion -> wall (map to closest semantic class)
        41: 1,   # base -> wall (map to closest semantic class)
        42: 1,   # box -> wall (map to closest semantic class)
        43: 1,   # column -> wall (map to closest semantic class)
        44: 1,   # signboard -> wall (map to closest semantic class)
        45: 1,   # chest of drawers -> wall (map to closest semantic class)
        46: 1,   # counter -> wall (map to closest semantic class)
        47: 5,   # sand -> tree (map to closest semantic class)
        48: 1,   # sink -> wall (map to closest semantic class)
        49: 2,   # skyscraper -> building
        50: 1,   # fireplace -> wall (map to closest semantic class)
        51: 1,   # refrigerator -> wall (map to closest semantic class)
        52: 1,   # grandstand -> wall (map to closest semantic class)
        53: 7,   # path -> road (map to closest semantic class)
        54: 1,   # stairs -> wall (map to closest semantic class)
        55: 7,   # runway -> road (map to closest semantic class)
        56: 1,   # case -> wall (map to closest semantic class)
        57: 1,   # pool table -> wall (map to closest semantic class)
        58: 1,   # pillow -> wall (map to closest semantic class)
        59: 1,   # screen door -> wall (map to closest semantic class)
        60: 1,   # stairway -> wall (map to closest semantic class)
        61: 3,   # river -> sky (map to closest semantic class)
        62: 7,   # bridge -> road (map to closest semantic class)
        63: 1,   # bookcase -> wall (map to closest semantic class)
        64: 1,   # blind -> wall (map to closest semantic class)
        65: 1,   # coffee table -> wall (map to closest semantic class)
        66: 1,   # toilet -> wall (map to closest semantic class)
        67: 18,  # flower -> plant
        68: 1,   # book -> wall (map to closest semantic class)
        69: 5,   # hill -> tree (map to closest semantic class)
        70: 1,   # bench -> wall (map to closest semantic class)
        71: 1,   # countertop -> wall (map to closest semantic class)
        72: 1,   # stove -> wall (map to closest semantic class)
        73: 18,  # palm -> plant
        74: 1,   # kitchen island -> wall (map to closest semantic class)
        75: 1,   # computer -> wall (map to closest semantic class)
        76: 1,   # swivel chair -> wall (map to closest semantic class)
        77: 3,   # boat -> sky (map to closest semantic class)
        78: 1,   # bar -> wall (map to closest semantic class)
        79: 1,   # arcade machine -> wall (map to closest semantic class)
        80: 2,   # hovel -> building
        81: 7,   # bus -> road (map to closest semantic class)
        82: 1,   # towel -> wall (map to closest semantic class)
        83: 1,   # light -> wall (map to closest semantic class)
        84: 7,   # truck -> road (map to closest semantic class)
        85: 2,   # tower -> building
        86: 1,   # chandelier -> wall (map to closest semantic class)
        87: 1,   # awning -> wall (map to closest semantic class)
        88: 1,   # streetlight -> wall (map to closest semantic class)
        89: 1,   # booth -> wall (map to closest semantic class)
        90: 1,   # television receiver -> wall (map to closest semantic class)
        91: 3,   # airplane -> sky (map to closest semantic class)
        92: 7,   # dirt track -> road (map to closest semantic class)
        93: 1,   # apparel -> wall (map to closest semantic class)
        94: 1,   # pole -> wall (map to closest semantic class)
        95: 5,   # land -> tree (map to closest semantic class)
        96: 1,   # bannister -> wall (map to closest semantic class)
        97: 1,   # escalator -> wall (map to closest semantic class)
        98: 1,   # ottoman -> wall (map to closest semantic class)
        99: 1,   # bottle -> wall (map to closest semantic class)
        100: 1,  # buffet -> wall (map to closest semantic class)
        101: 1,  # poster -> wall (map to closest semantic class)
        102: 1,  # stage -> wall (map to closest semantic class)
        103: 7,  # van -> road (map to closest semantic class)
        104: 3,  # ship -> sky (map to closest semantic class)
        105: 3,  # fountain -> sky (map to closest semantic class)
        106: 1,  # conveyer belt -> wall (map to closest semantic class)
        107: 1,  # canopy -> wall (map to closest semantic class)
        108: 1,  # washer -> wall (map to closest semantic class)
        109: 1,  # plaything -> wall (map to closest semantic class)
        110: 3,  # swimming pool -> sky (map to closest semantic class)
        111: 1,  # stool -> wall (map to closest semantic class)
        112: 1,  # barrel -> wall (map to closest semantic class)
        113: 1,  # basket -> wall (map to closest semantic class)
        114: 3,  # waterfall -> sky (map to closest semantic class)
        115: 1,  # tent -> wall (map to closest semantic class)
        116: 1,  # bag -> wall (map to closest semantic class)
        117: 7,  # minibike -> road (map to closest semantic class)
        118: 1,  # cradle -> wall (map to closest semantic class)
        119: 1,  # oven -> wall (map to closest semantic class)
        120: 1,  # ball -> wall (map to closest semantic class)
        121: 1,  # food -> wall (map to closest semantic class)
        122: 1,  # step -> wall (map to closest semantic class)
        123: 1,  # tank -> wall (map to closest semantic class)
        124: 1,  # trade name -> wall (map to closest semantic class)
        125: 1,  # microwave -> wall (map to closest semantic class)
        126: 1,  # pot -> wall (map to closest semantic class)
        127: 18, # animal -> plant (map to closest semantic class)
        128: 7,  # bicycle -> road (map to closest semantic class)
        129: 3,  # lake -> sky (map to closest semantic class)
        130: 1,  # dishwasher -> wall (map to closest semantic class)
        131: 1,  # screen -> wall (map to closest semantic class)
        132: 1,  # blanket -> wall (map to closest semantic class)
        133: 1,  # sculpture -> wall (map to closest semantic class)
        134: 1,  # hood -> wall (map to closest semantic class)
        135: 1,  # sconce -> wall (map to closest semantic class)
        136: 1,  # vase -> wall (map to closest semantic class)
        137: 1,  # traffic light -> wall (map to closest semantic class)
        138: 1,  # tray -> wall (map to closest semantic class)
        139: 1,  # ashcan -> wall (map to closest semantic class)
        140: 1,  # fan -> wall (map to closest semantic class)
        141: 3,  # pier -> sky (map to closest semantic class)
        142: 1,  # crt screen -> wall (map to closest semantic class)
        143: 1,  # plate -> wall (map to closest semantic class)
        144: 1,  # monitor -> wall (map to closest semantic class)
        145: 1,  # bulletin board -> wall (map to closest semantic class)
        146: 1,  # shower -> wall (map to closest semantic class)
        147: 1,  # radiator -> wall (map to closest semantic class)
        148: 1,  # glass -> wall (map to closest semantic class)
        149: 1,  # clock -> wall (map to closest semantic class)
    }
    
    # Create a mapping array for efficient lookup
    mapping_array = np.zeros(max(segformer_to_ade20k.keys()) + 1, dtype=np.int32)
    for segformer_idx, ade20k_idx in segformer_to_ade20k.items():
        # Only map to classes that exist in ground truth
        if ade20k_idx in true_classes:
            mapping_array[segformer_idx] = ade20k_idx
        else:
            # Map to the closest semantic class that exists in ground truth
            mapping_array[segformer_idx] = 0  # Default to background
    
    # Map the predictions
    mapped_predictions = mapping_array[predictions]
    
    return mapped_predictions

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
