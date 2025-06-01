import os
import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path

# ADE20K dataset configuration
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
    Load ADE20K dataset from local directory.
    Returns a dataset of (image, mask) pairs.
    """
    print("📥 Loading ADE20K dataset...")
    
    data_dir = Path(data_dir)
    val_dir = data_dir / "validation"
    
    if not val_dir.exists():
        raise ValueError(f"Validation directory not found: {val_dir}")
    
    # Get list of image files
    image_files = sorted((val_dir / "images").glob("*.jpg"))
    mask_files = sorted((val_dir / "annotations").glob("*.png"))
    
    if not image_files or not mask_files:
        raise ValueError("No images or masks found in validation directory")
    
    print(f"Found {len(image_files)} validation images")
    
    def load_and_preprocess(image_path, mask_path):
        # Load image
        image = tf.io.read_file(str(image_path))
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, ADE20K_CONFIG['input_size'])
        image = tf.cast(image, tf.float32) / 255.0
        
        # Load mask
        mask = tf.io.read_file(str(mask_path))
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
    Returns the ADE20K color mapping for visualization.
    Each class has a unique RGB color.
    """
    # Generate a deterministic colormap
    np.random.seed(42)
    colormap = np.random.randint(0, 255, size=(ADE20K_CONFIG['num_classes'], 3), dtype=np.uint8)
    # Set background (class 0) to black
    colormap[0] = [0, 0, 0]
    return colormap

def colorize_ade20k_mask(mask):
    """
    Convert a segmentation mask to a color image using ADE20K colormap.
    
    Args:
        mask: Segmentation mask (H, W) with class indices
        
    Returns:
        Color image (H, W, 3) with RGB values
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
    Save original image, true mask, and predicted mask side by side.
    
    Args:
        image: Original image (H, W, 3)
        true_mask: Ground truth mask (H, W)
        pred_mask: Predicted mask (H, W)
        output_dir: Directory to save results
        index: Image index for filename
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