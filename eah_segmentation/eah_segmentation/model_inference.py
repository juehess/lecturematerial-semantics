# eah_segmentation/model_inference.py

import cv2
import numpy as np

def preprocess_image(image_bgr, input_size=(512, 512)):
    """
    Preprocess image for Cityscapes-trained models
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, input_size)
    normalized = resized.astype(np.float32) / 255.0
    return normalized

def run_inference_on_image(model, image_bgr, input_size=(512, 512)):
    """
    Run inference with a Cityscapes-trained Keras model
    
    Args:
        model: Keras model object
        image_bgr: Input image in BGR format
        input_size: Tuple of (width, height) for resizing
    """
    input_tensor = preprocess_image(image_bgr, input_size=input_size)
    input_tensor = np.expand_dims(input_tensor, axis=0)  # shape (1, H, W, 3)
    predictions = model.predict(input_tensor)            # shape (1, H, W, num_classes)
    seg_mask = np.argmax(predictions[0], axis=-1)       # shape (H, W)
    return seg_mask
