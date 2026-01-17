import os
import tensorflow as tf

# --- Configuration ---
# This dictionary has been updated to match your new filenames.
# It maps your final .keras model files to their desired .tflite output names.

MODELS_TO_CONVERT = {
    # 1. The Gatekeeper Model
    'cat_vs_not_cat.keras': 'gatekeeper_model.tflite',
    
    # 2. The Generalist 12-Breed Model
    'breed_expert_mobilenetV3.keras': 'generalist_breed_model.tflite',

    # 3. The Specialized 5-Breed Expert Model
    'similar_breed.keras': 'similar_breed_expert_model.tflite'
}

def convert_model_to_tflite(keras_model_path, tflite_model_path):
    """
    Loads a .keras model and converts it into a .tflite model.
    """
    print(f"\n--- Converting '{keras_model_path}' ---")

    if not os.path.exists(keras_model_path):
        print(f"❌ ERROR: Model file not found at '{keras_model_path}'.")
        print("   Please check that the filename is correct and it's in the same directory.")
        print("   Skipping this conversion.")
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
    print("--- Starting TensorFlow Lite Conversion Process for All Models ---")
    for keras_file, tflite_file in MODELS_TO_CONVERT.items():
        convert_model_to_tflite(keras_file, tflite_file)
    print("\n--- All conversions complete. ---")

if __name__ == '__main__':
    main()
