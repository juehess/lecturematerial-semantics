"""Simple model name mapping configuration."""

# Maps simple model names to their full filenames
MODEL_NAMES = {
    'segformer': 'segformer_b0',
    'deeplabv3plus': 'deeplabv3plus_edgetpu',
    'mosaic': 'mosaic',
    'deeplabv3edge': 'deeplabv3_cityscapes',
    'deeplabv3edge': 'deeplabv3_cityscapes'  # Same directory, different model file
} 