# eah_segmentation/color_utils.py

import numpy as np

"""
This module provides utilities for colorizing segmentation masks using
a modified Cityscapes color palette that better matches ADE20K colors.
It's primarily used for visualizing semantic segmentation predictions.

Key Features:
- Implements modified Cityscapes color scheme to match ADE20K
- Efficient vectorized color mapping
- Handles invalid class indices
"""

def colorize_mask(seg_mask):
    """
    Converts a segmentation mask to a colored visualization using modified colors
    that better match ADE20K's color scheme.
    
    Color Palette:
        - Background: Black (0, 0, 0)
        - Road: Gray (128, 128, 128)  # More similar to ADE20K road
        - Sidewalk: Light Gray (200, 200, 200)  # More similar to ADE20K sidewalk
        - Building: Brown (128, 64, 64)  # More similar to ADE20K building
        - Wall: Dark Gray (70, 70, 70)  # Matches ADE20K wall
        - And 15 more classes...
    
    Key Features:
        - Vectorized implementation for efficiency
        - Clips invalid class indices
        - Uses colors more aligned with ADE20K palette
    
    Args:
        seg_mask (np.ndarray): Segmentation mask of shape (H, W) with class indices
        
    Returns:
        np.ndarray: Colored mask of shape (H, W, 3) with RGB values
    """
    # Modified color palette to better match ADE20K colors
    palette = [
        (0, 0, 0),        # background -> black (same)
        (128, 128, 128),  # road -> gray (more like ADE20K)
        (200, 200, 200),  # sidewalk -> light gray (more like ADE20K)
        (128, 64, 64),    # building -> brown (more like ADE20K)
        (70, 70, 70),     # wall -> dark gray (matches ADE20K)
        (153, 153, 153),  # fence -> gray (similar to ADE20K)
        (128, 128, 0),    # pole -> yellow-gray (more like ADE20K)
        (250, 170, 30),   # traffic light -> orange (kept distinctive)
        (220, 220, 0),    # traffic sign -> yellow (kept distinctive)
        (107, 142, 35),   # vegetation -> green (matches ADE20K)
        (152, 251, 152),  # terrain -> light green (similar to ADE20K)
        (70, 130, 180),   # sky -> blue (matches ADE20K)
        (220, 20, 60),    # person -> red (matches ADE20K)
        (255, 0, 0),      # rider -> bright red (similar to ADE20K person)
        (0, 0, 142),      # car -> dark blue (more like ADE20K)
        (0, 0, 70),       # truck -> darker blue (variation of vehicle)
        (0, 60, 100),     # bus -> blue (variation of vehicle)
        (0, 80, 100),     # train -> blue (variation of vehicle)
        (0, 0, 230),      # motorcycle -> bright blue (variation of vehicle)
        (119, 11, 32)     # bicycle -> dark red (variation of vehicle)
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
