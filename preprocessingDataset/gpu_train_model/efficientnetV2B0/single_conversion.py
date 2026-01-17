import os
import tensorflow as tf
import argparse # We'll use argparse to read command-line arguments

def convert_model_to_tflite(keras_model_path, tflite_model_path):
    """Convert one Keras model file into a TFLite file.

    Inputs: paths for the .keras file and the .tflite output.
    """
    print(f"\n--- Converting '{keras_model_path}' ---")

    if not os.path.exists(keras_model_path):
        print(f"❌ ERROR: Model file not found at '{keras_model_path}'.")
        print("   Please check the filename and path.")
        return

    try:
        # Load the Keras model
        print("  - Loading Keras model...")
        model = tf.keras.models.load_model(keras_model_path, compile=False)

        # Initialize the TFLite converter
        print("  - Initializing TFLite converter...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Apply default optimizations for size and speed
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Perform the conversion
        print("  - Converting to TensorFlow Lite...")
        tflite_model = converter.convert()

        # Save the .tflite model
        with open(tflite_model_path, 'wb') as f:
            f.write(tflite_model)
        
        original_size = os.path.getsize(keras_model_path) / (1024 * 1024)
        tflite_size = len(tflite_model) / (1024 * 1024)
        
        print(f"  - ✅ Conversion successful!")
        print(f"  - Original size: {original_size:.2f} MB")
        print(f"  - TFLite size:   {tflite_size:.2f} MB")
        print(f"  - Saved as '{tflite_model_path}'")

    except Exception as e:
        print(f"  - ❌ ERROR during conversion: {e}")

def main():
    parser = argparse.ArgumentParser(description="Convert one .keras model to .tflite")

    # Input .keras model path (required)
    parser.add_argument("input_model", type=str, help="Path to the .keras model file")

    # Output .tflite path (required)
    parser.add_argument("output_model", type=str, help="Output .tflite file path")
    
    args = parser.parse_args()

    # Run the conversion function with the provided arguments
    convert_model_to_tflite(args.input_model, args.output_model)

if __name__ == '__main__':
    main()