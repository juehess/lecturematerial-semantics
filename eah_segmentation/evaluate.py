"""Command-line interface for model evaluation."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from pathlib import Path
import argparse
from datetime import datetime

try:
    from ade20k_utils import load_ade20k_dataset
    from utils import load_model, save_timing_results
    from metrics import evaluate_model
    from model_config import MODEL_NAMES
except ImportError:
    from eah_segmentation.ade20k_utils import load_ade20k_dataset
    from eah_segmentation.utils import load_model, save_timing_results
    from eah_segmentation.metrics import evaluate_model
    from eah_segmentation.model_config import MODEL_NAMES

def main():
    parser = argparse.ArgumentParser(description='Evaluate semantic segmentation models.')
    parser.add_argument('--models', nargs='+', required=True,
                      help='Model names to evaluate')
    parser.add_argument('--model_type', choices=['keras', 'tflite'], default='keras',
                      help='Type of model to use (tflite or keras)')
    parser.add_argument('--data_dir', type=str, default='datasets',
                      help='Directory containing ADE20K dataset')
    parser.add_argument('--num_images', type=int, default=1,
                      help='Number of images to test on (default: 1)')
    parser.add_argument('--image_index', type=int, default=0,
                      help='Index of the image to test (default: 0)')
    parser.add_argument('--device', type=str, choices=['cpu', 'coral'], default='cpu',
                      help='Device to run inference on (cpu or coral)')
    parser.add_argument('--debug', type=lambda x: x.lower() == 'true', choices=[True, False], default=False,
                      help='Enable debug mode with additional logging')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    # Load ADE20K dataset
    print(f"📥 Loading ADE20K dataset from {args.data_dir}")
    dataset = load_ade20k_dataset(args.data_dir)
    
    # Skip to the desired image index
    dataset = dataset.skip(args.image_index)
    
    # Store timing results for all models
    all_results = []
    
    # Test each model
    for model_name in args.models:
        # Create model-specific output directory
        model_output_dir = output_dir / f"{model_name}_{args.model_type}_{args.device}"
        model_output_dir.mkdir(exist_ok=True)
        
        try:
            # Load model
            model, load_time = load_model(model_name, args.model_type, args.device)
            
            # Evaluate model
            results = evaluate_model(model, dataset, model_output_dir, args.num_images, model_name, debug=args.debug)
            results['load_time'] = load_time
            
            # Save individual model results
            save_timing_results(results, model_output_dir)
            
            # Add to combined results
            all_results.append(results)
            
        except Exception as e:
            print(f"❌ Error testing {model_name}: {str(e)}")
            continue
    
    # Save combined results
    if all_results:
        combined_results = {
            'timestamp': datetime.now().isoformat(),
            'models': all_results
        }
        save_timing_results(combined_results, output_dir)

if __name__ == '__main__':
    main() 