import os
import tensorflow as tf
import numpy as np
from pathlib import Path
import argparse

# Try relative imports first, fall back to absolute imports
try:
    from .ade20k_utils import load_ade20k_dataset, save_prediction
    from .model_inference import run_inference_on_image
except ImportError:
    from ade20k_utils import load_ade20k_dataset, save_prediction
    from model_inference import run_inference_on_image

def evaluate_model(model, dataset, output_dir, num_images=10, model_name=None):
    """
    Evaluate a model on ADE20K dataset and save visualizations.
    
    Args:
        model: Loaded model
        dataset: ADE20K dataset
        output_dir: Directory to save results
        num_images: Number of images to evaluate
        model_name: Name of the model (e.g., 'segformer_b0', 'deeplabv3plus_edgetpu', 'mosaic')
    """
    print(f"\n🔍 Evaluating model on {num_images} image{'s' if num_images > 1 else ''}...")
    
    for i, (image, true_mask) in enumerate(dataset):
        if i >= num_images:
            break
            
        # Run inference
        print(f"📸 Processing image {i+1}...")
        
        # Get ground truth classes
        true_classes = np.unique(true_mask.numpy()[0])
        print("\n📊 Ground truth classes:")
        for cls in true_classes:
            print(f"  Class {cls}")
        
        # Run inference with ground truth classes
        pred_mask = run_inference_on_image(model, image.numpy()[0], model_name, true_classes)
        
        # Save visualization
        save_prediction(
            image.numpy()[0],
            true_mask.numpy()[0],
            pred_mask,
            output_dir,
            i
        )
        
        print(f"✅ Processed image {i+1}/{num_images}")

def main():
    parser = argparse.ArgumentParser(description='Test segmentation models on ADE20K dataset')
    parser.add_argument('--models', type=str, nargs='+', required=True,
                      help='List of model names to test')
    parser.add_argument('--model_type', type=str, choices=['tflite', 'keras'], default='keras',
                      help='Type of model to use (tflite or keras)')
    parser.add_argument('--num_images', type=int, default=1,
                      help='Number of images to test on (default: 1)')
    parser.add_argument('--output_dir', type=str, default='results',
                      help='Directory to save results')
    parser.add_argument('--data_dir', type=str, default='datasets',
                      help='Directory containing ADE20K dataset')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load ADE20K dataset
    print(f"📥 Loading ADE20K dataset from {args.data_dir}")
    dataset = load_ade20k_dataset(args.data_dir)
    
    # Test each model
    for model_name in args.models:
        print(f"\n🚀 Testing {model_name} ({args.model_type})...")
        
        # Create model-specific output directory
        model_output_dir = output_dir / f"{model_name}_{args.model_type}"
        model_output_dir.mkdir(exist_ok=True)
        
        # Load model
        model_dir = Path(__file__).parent.parent / 'models' / model_name
        if not model_dir.exists():
            print(f"❌ Model directory not found: {model_dir}")
            continue
            
        try:
            if args.model_type == 'tflite':
                # Load TFLite model
                tflite_path = model_dir / 'tflite' / '1.tflite'
                if tflite_path.exists():
                    print(f"📥 Loading TFLite model from {tflite_path}")
                    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
                    interpreter.allocate_tensors()
                    model = interpreter
                else:
                    print(f"❌ TFLite model not found: {tflite_path}")
                    continue
            else:
                # Load Keras model
                keras_path = model_dir / 'keras'
                if keras_path.exists():
                    print(f"📥 Loading Keras model from {keras_path}")
                    model = tf.saved_model.load(str(keras_path))
                else:
                    print(f"❌ Keras model not found: {keras_path}")
                    continue
            
            # Evaluate model
            evaluate_model(model, dataset, model_output_dir, args.num_images, model_name)
            
        except Exception as e:
            print(f"❌ Error testing {model_name}: {str(e)}")
            continue

if __name__ == '__main__':
    main() 