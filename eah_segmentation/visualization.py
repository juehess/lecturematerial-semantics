# eah_segmentation/color_utils.py

import numpy as np

"""
This module provides utilities for colorizing segmentation masks using
the Cityscapes color palette. It's primarily used for visualizing
semantic segmentation predictions.

Key Features:
- Implements Cityscapes color scheme
- Efficient vectorized color mapping
- Handles invalid class indices
"""

def colorize_mask(seg_mask):
    """
    Converts a segmentation mask to a colored visualization using Cityscapes colors.
    
    Color Palette:
        - Background: Black (0, 0, 0)
        - Road: Purple-Gray (128, 64, 128)
        - Sidewalk: Pink (244, 35, 232)
        - Building: Dark Gray (70, 70, 70)
        - Wall: Blue-Gray (102, 102, 156)
        - And 15 more classes...
    
    Key Features:
        - Vectorized implementation for efficiency
        - Clips invalid class indices
        - Uses standard Cityscapes colors
    
    Args:
        seg_mask (np.ndarray): Segmentation mask of shape (H, W) with class indices
        
    Returns:
        np.ndarray: Colored mask of shape (H, W, 3) with RGB values
    """
    # Cityscapes color palette
    palette = [
        (0, 0, 0),        # background
        (128, 64, 128),   # road
        (244, 35, 232),   # sidewalk
        (70, 70, 70),     # building
        (102, 102, 156),  # wall
        (190, 153, 153),  # fence
        (153, 153, 153),  # pole
        (250, 170, 30),   # traffic light
        (220, 220, 0),    # traffic sign
        (107, 142, 35),   # vegetation
        (152, 251, 152),  # terrain
        (70, 130, 180),   # sky
        (220, 20, 60),    # person
        (255, 0, 0),      # rider
        (0, 0, 142),      # car
        (0, 0, 70),       # truck
        (0, 60, 100),     # bus
        (0, 80, 100),     # train
        (0, 0, 230),      # motorcycle
        (119, 11, 32)     # bicycle
    ]

    h, w = seg_mask.shape
    color_image = np.zeros((h, w, 3), dtype=np.uint8)

    # Clip the segmentation mask to valid class indices
    seg_mask_clipped = np.clip(seg_mask, 0, len(palette)-1)

    # Vectorized color mapping
    for class_idx, color in enumerate(palette):
        mask = seg_mask_clipped == class_idx
        color_image[mask] = color

    return color_image
