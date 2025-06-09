import os
import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path

"""
eah_segmentation/ade20k_utils.py

This module provides utilities for working with the ADE20K dataset,
including dataset loading, preprocessing, and visualization functions.

Key Components:
- ADE20K dataset configuration (classes, names, sizes)
- Dataset loading and preprocessing pipeline
- Visualization utilities for segmentation masks
"""

# Constants
ADE20K_CONFIG = {
    'num_classes': 150,
    'input_size': (512, 512),
    'class_names': [
        'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed',
        'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth', 'door',
        'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water', 'painting',
        'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair', 'seat',
        'fence', 'desk', 'rock', 'wardrobe', 'lamp', 'bathtub', 'railing', 'cushion',
        'base', 'box', 'column', 'signboard', 'chest of drawers', 'counter', 'sand',
        'sink', 'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path',
        'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door', 'stairway',
        'river', 'bridge', 'bookcase', 'blind', 'coffee table', 'toilet', 'flower',
        'book', 'hill', 'bench', 'countertop', 'stove', 'palm', 'kitchen island',
        'computer', 'swivel chair', 'boat', 'bar', 'arcade machine', 'hovel', 'bus',
        'towel', 'light', 'truck', 'tower', 'chandelier', 'awning', 'streetlight',
        'booth', 'television receiver', 'airplane', 'dirt track', 'apparel', 'pole',
        'land', 'bannister', 'escalator', 'ottoman', 'bottle', 'buffet', 'poster',
        'stage', 'van', 'ship', 'fountain', 'conveyer belt', 'canopy', 'washer',
        'plaything', 'swimming pool', 'stool', 'barrel', 'basket', 'waterfall',
        'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank',
        'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher',
        'screen', 'blanket', 'sculpture', 'hood', 'sconce', 'vase', 'traffic light',
        'tray', 'ashcan', 'fan', 'pier', 'crt screen', 'plate', 'monitor', 'bulletin board',
        'shower', 'radiator', 'glass', 'clock', 'flag'
    ]
}

def load_ade20k_dataset(data_dir, batch_size=1):
    """
    Loads and preprocesses the ADE20K dataset.
    
    Key Operations:
        1. Loads image and mask pairs
        2. Resizes to standard input size
        3. Normalizes image values
        4. Creates TensorFlow dataset pipeline
        
    Args:
        data_dir (str/Path): Directory containing ADE20K dataset
        batch_size (int): Batch size for dataset
        
    Returns:
        tf.data.Dataset: Dataset yielding (image, mask) pairs
    """
    print("📥 Loading ADE20K dataset...")
    
    data_dir = Path(data_dir)
    ade_dir = data_dir / "ADEChallengeData2016"
    
    if not ade_dir.exists():
        raise ValueError(f"ADE20K directory not found: {ade_dir}")
    
    # Get list of image files
    image_files = sorted((ade_dir / "images" / "validation").glob("*.jpg"))
    mask_files = sorted((ade_dir / "annotations" / "validation").glob("*.png"))
    
    if not image_files or not mask_files:
        raise ValueError("No images or masks found in validation directory")
    
    print(f"Found {len(image_files)} validation images")
    
    # Convert Path objects to strings
    image_files = [str(f) for f in image_files]
    mask_files = [str(f) for f in mask_files]
    
    def load_and_preprocess(image_path, mask_path):
        # Load image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, ADE20K_CONFIG['input_size'])
        image = tf.cast(image, tf.float32) / 255.0
        
        # Load mask
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.image.resize(mask, ADE20K_CONFIG['input_size'], 
                             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        mask = tf.squeeze(mask)
        
        return image, mask
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_files, mask_files))
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def get_ade20k_colormap():
    """
    Generates a deterministic color mapping for ADE20K classes.
    
    Features:
        - Uses fixed random seed for consistency
        - Maps each class to unique RGB color
        - Sets background (class 0) to black
        
    Returns:
        np.ndarray: Array of shape (num_classes, 3) containing RGB values
    """
    # Generate a deterministic colormap
    np.random.seed(42)
    colormap = np.random.randint(0, 255, size=(ADE20K_CONFIG['num_classes'], 3), dtype=np.uint8)
    # Set background (class 0) to black
    colormap[0] = [0, 0, 0]
    return colormap

def colorize_ade20k_mask(mask):
    """
    Converts a segmentation mask to a colored visualization.
    
    Args:
        mask (np.ndarray): Segmentation mask of shape (H, W) or (H, W, num_classes)
        
    Returns:
        np.ndarray: Colored mask of shape (H, W, 3)
    """
    colormap = get_ade20k_colormap()
    
    # Ensure mask is 2D
    if len(mask.shape) == 3:
        mask = np.argmax(mask, axis=-1)
    
    # Create color mask
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for i in range(ADE20K_CONFIG['num_classes']):
        color_mask[mask == i] = colormap[i]
    
    return color_mask

def save_prediction(image, true_mask, pred_mask, output_dir, index):
    """
    Saves a side-by-side visualization of segmentation results.
    
    Creates a visualization containing:
    - Original image
    - Ground truth segmentation mask
    - Predicted segmentation mask
    
    Args:
        image (np.ndarray): Original image (H, W, 3)
        true_mask (np.ndarray): Ground truth mask (H, W)
        pred_mask (np.ndarray): Predicted mask (H, W)
        output_dir (str): Directory to save visualization
        index (int): Image index for filename
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert masks to color images
    true_color = colorize_ade20k_mask(true_mask)
    pred_color = colorize_ade20k_mask(pred_mask)
    
    # Convert image to uint8
    image = (image * 255).astype(np.uint8)
    
    # Create side-by-side visualization
    vis = np.hstack([image, true_color, pred_color])
    
    # Save visualization
    output_path = os.path.join(output_dir, f'prediction_{index:04d}.png')
    cv2.imwrite(output_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"💾 Saved visualization to {output_path}")