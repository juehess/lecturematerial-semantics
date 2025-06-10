"""EAH Semantic Segmentation package."""

from .utils import load_model
from .model_config import MODEL_NAMES
from .inference import run_inference_on_image
from .visualization import colorize_mask
from .class_mapping import (
    map_segformer_to_ade20k,
    map_mosaic_to_cityscapes,
    map_cityscapes_to_ade20k
)
from .ade20k_utils import load_ade20k_dataset, save_prediction

__all__ = [
    'run_inference_on_image',
    'colorize_mask',
    'load_model',
    'MODEL_NAMES',
    'map_segformer_to_ade20k',
    'map_mosaic_to_cityscapes',
    'map_cityscapes_to_ade20k',
    'load_ade20k_dataset',
    'save_prediction'
]
