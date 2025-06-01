import os
import argparse
import tensorflow as tf
import numpy as np
from PIL import Image

def load_images_from_directory(directory, image_size=(512, 512), max_images=100):
    images = []
    for i, filename in enumerate(sorted(os.listdir(directory))):
        if i >= max_images:
            break
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(directory, filename)
            image = Image.open(image_path).convert("RGB")
            image = image.resize(image_size)
            image = np.asarray(image).astype(np.float32) / 255.0
            images.append(image)
    return images

def representative_data_gen(images):
    for img in images:
        img = np.expand_dims(img, axis=0)
        yield [img]

def convert_keras_to_tflite(model_path, image_dir, output_path=None, image_size=(512, 512)):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    print("📥 Loading model...")
    model = tf.keras.models.load_model(model_path)

    print("🖼 Loading representative dataset...")
    images = load_images_from_directory(image_dir, image_size=image_size)

    print("🔄 Converting model to quantized TFLite format...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_data_gen(images)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()

    if output_path is None:
        base, _ = os.path.splitext(model_path)
        output_path = base + "_quant.tflite"

    print(f"💾 Saving TFLite model to {output_path}...")
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    print(f"✅ Quantized TFLite model saved: {output_path} ({len(tflite_model)/1024/1024:.2f} MB)")

def main():
    parser = argparse.ArgumentParser(description="Convert any Keras model to quantized TFLite using a representative dataset.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the .keras model file")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing representative images")
    parser.add_argument("--output_path", type=str, default=None, help="Optional output path for the TFLite model")
    parser.add_argument("--image_size", type=int, nargs=2, default=(512, 512), help="Input image size (width height)")

    args = parser.parse_args()
    convert_keras_to_tflite(args.model_path, args.image_dir, args.output_path, tuple(args.image_size))

if __name__ == "__main__":
    main()