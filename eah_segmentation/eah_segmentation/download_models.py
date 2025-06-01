import os
import shutil
import kagglehub
import tensorflow as tf
import numpy as np
import keras_cv
import keras_hub

# Model aliases and their Kaggle Hub paths
MODEL_PATHS = {
    "deeplabv3plus_edgetpu": {
        "keras": "google/deeplab-edgetpu/tensorFlow2/default-argmax-m/1",
        "tflite": "google/deeplab-edgetpu/tfLite/default-argmax-m/1"
    },
    "mosaic": {
        "tflite": "google/mosaic/tfLite/mobilenetmultiavgseg/1"
    }
}

# Local base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

def download_model(name, model_paths, model_dir):
    """Download model files from Kaggle Hub and organize them in a consistent structure"""
    print(f"\n🔄 Downloading {name} from Kaggle Hub...")
    
    # Create model-specific directory
    model_specific_dir = os.path.join(model_dir, name)
    os.makedirs(model_specific_dir, exist_ok=True)
    
    try:
        # Download each available format
        for format_type, kaggle_path in model_paths.items():
            print(f"📥 Downloading {format_type} format...")
            model_src = kagglehub.model_download(kaggle_path)
            print(f"✅ Downloaded {format_type} to {model_src}")

            # Create format-specific directory
            format_dir = os.path.join(model_specific_dir, format_type)
            if os.path.exists(format_dir):
                print(f"✅ {format_type} already exists at {format_dir}, skipping copy.")
            else:
                shutil.copytree(model_src, format_dir)
                print(f"✅ Copied {format_type} to {format_dir}")
                
    except Exception as e:
        print(f"❌ Failed to process {name}: {e}")

def convert_segformer_to_tflite(model_dir: str):
    """
    Convert SegFormer model from KaggleHub to TFLite format.
    This function:
    1. Loads the SegFormer model from KaggleHub
    2. Exports it as a TensorFlow SavedModel
    3. Converts it to TFLite format
    4. Saves the TFLite model
    """
    print("\n🚀 Starting SegFormer B0 conversion process...")
    print("📥 Loading SegFormer B0 model from KaggleHub (ADE20K preset)...")

    try:
        # Load model from KaggleHub
        model = keras_hub.models.SegFormerImageSegmenter.from_preset(
            "kaggle://keras/segformer/keras/segformer_b0_ade20k_512"
        )
        print("✅ Successfully loaded SegFormer model")

        # Create model-specific directory
        model_specific_dir = os.path.join(model_dir, "segformer_b0")
        os.makedirs(model_specific_dir, exist_ok=True)

        # Export model to SavedModel directory
        saved_model_path = os.path.join(model_specific_dir, "keras")
        print(f"💾 Exporting SavedModel to {saved_model_path}...")
        model.export(saved_model_path)
        print("✅ SavedModel exported successfully")

        # Convert to TFLite
        print("🔄 Converting to TFLite format...")
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        print("✅ TFLite conversion completed")

        # Save TFLite model
        tflite_path = os.path.join(model_specific_dir, "tflite", "1.tflite")
        print(f"💾 Saving TFLite model to {tflite_path}...")
        os.makedirs(os.path.dirname(tflite_path), exist_ok=True)
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        print("✅ TFLite model saved successfully")
        print(f"📊 TFLite model size: {len(tflite_model) / (1024 * 1024):.2f} MB")

    except Exception as e:
        print(f"❌ Error during SegFormer conversion: {str(e)}")
        raise

def main():
    print("🚀 Starting Kaggle Hub model downloads...")
    os.makedirs(MODEL_DIR, exist_ok=True)

    for name, model_paths in MODEL_PATHS.items():
        download_model(name, model_paths, MODEL_DIR)

    print("\n🚀 Building and converting SegFormer B0...")
    convert_segformer_to_tflite(MODEL_DIR)

    print("\n🎉 All models downloaded and converted successfully.")

if __name__ == "__main__":
    main()
