"""Class mapping utilities for different segmentation models."""

import numpy as np

def map_segformer_to_ade20k(predictions, true_classes=None):
    """
    Maps SegFormer class indices to ADE20K class indices.
    
    Args:
        predictions (np.ndarray): Model predictions with class indices
        true_classes (list, optional): List of ground truth class indices
        
    Returns:
        np.ndarray: Predictions mapped to ADE20K class indices
    """
    # Simply add 1 to all predictions to match ADE20K's 1-based indexing
    return predictions + 1

def map_mosaic_to_cityscapes(predictions):
    """
    Maps Mosaic model predictions to Cityscapes class indices.
    
    Args:
        predictions (np.ndarray): Raw model predictions
        
    Returns:
        np.ndarray: Predictions mapped to Cityscapes classes
    """
    # Cityscapes classes
    cityscapes_classes = {
        0: 0,     # background/unlabeled
        1: 1,     # road
        2: 2,     # sidewalk
        3: 3,     # building
        4: 4,     # wall
        5: 5,     # fence
        6: 6,     # pole
        7: 7,     # traffic light
        8: 8,     # traffic sign
        9: 9,     # vegetation
        10: 10,   # terrain
        11: 11,   # sky
        12: 12,   # person
        13: 13,   # rider
        14: 14,   # car
        15: 15,   # truck
        16: 16,   # bus
        17: 17,   # train
        18: 18,   # motorcycle
        19: 19    # bicycle
    }
    
    # Create mapping array
    mapping = np.zeros(np.max(predictions) + 1, dtype=np.int32)
    for mosaic_idx, cityscapes_idx in cityscapes_classes.items():
        mapping[mosaic_idx] = cityscapes_idx
    
    return mapping[predictions]

def map_cityscapes_to_ade20k(predictions):
    """
    Maps Cityscapes class indices to ADE20K class indices.
    
    Args:
        predictions (np.ndarray): Predictions with Cityscapes class indices
        
    Returns:
        np.ndarray: Predictions mapped to ADE20K class indices
    """
    # Mapping from Cityscapes to ADE20K classes
    mapping = {
        0: 0,    # void -> background
        1: 2,    # road -> road
        2: 3,    # sidewalk -> sidewalk
        3: 1,    # building -> building
        4: 4,    # wall -> wall
        5: 5,    # fence -> fence
        6: 17,   # pole -> pole
        7: 19,   # traffic light -> traffic light
        8: 20,   # traffic sign -> traffic sign
        9: 21,   # vegetation -> vegetation
        10: 22,  # terrain -> terrain
        11: 23,  # sky -> sky
        12: 24,  # person -> person
        13: 25,  # rider -> rider
        14: 26,  # car -> car
        15: 27,  # truck -> truck
        16: 28,  # bus -> bus
        17: 29,  # train -> train
        18: 30,  # motorcycle -> motorcycle
        19: 31   # bicycle -> bicycle
    }
    
    # Create mapping array
    max_idx = max(mapping.keys())
    mapping_array = np.zeros(max_idx + 1, dtype=np.int32)
    for cityscapes_idx, ade20k_idx in mapping.items():
        mapping_array[cityscapes_idx] = ade20k_idx
    
    return mapping_array[predictions] 