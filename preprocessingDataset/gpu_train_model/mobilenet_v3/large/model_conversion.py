import os
import shutil
import tensorflow as tf

# ======================================================
# CONFIGURATION
# ======================================================

# Toggle this ON/OFF if want smaller + faster models
# True  = Use TFLite optimization (quantization, pruning)
# False = Keep full-precision float32 models (maximum accuracy)
ENABLE_OPTIMIZATION = True

# List of models to convert
MODELS_TO_CONVERT = {  
    #'cat_vs_not_cat.keras': 'gatekeeper_model.tflite', # The Gatekeeper Model (99.94% Accuracy)
    'breed_expert_mobilenetv3_large_v1.keras': 'generalist_breed_model.tflite', # The Generalist 12-Breed Model (93.47% Accuracy)
    #'similar_breed_expert_effnet_v1.keras': 'similar_breed_expert_model.tflite'# The Specialized 5-Breed Expert Model (95.50% Accuracy)
}

# ======================================================
# MODEL CONVERSION FUNCTION
# ======================================================
def convert_model_to_tflite(keras_model_path, tflite_model_path):
    """
    Converts a .keras model → .tflite format.
    Handles modern (TensorFlow 2.15+) export process, temporary cleanup,
    and optional optimization for size/speed.
    """
    print(f"\n--- Converting '{keras_model_path}' ---")

    full_in = os.path.abspath(keras_model_path)
    full_out = os.path.join(os.path.dirname(__file__), tflite_model_path)
    tmp_folder_name = f"tmp_saved_{os.path.splitext(keras_model_path)[0]}"
    saved_dir = os.path.join(os.path.dirname(__file__), tmp_folder_name)

    print(f"  - Input Model:  {full_in}")
    print(f"  - Output TFLite: {full_out}")

    if not os.path.exists(full_in):
        print("ERROR: File not found.")
        return

    try:
        print("  [1] Loading Keras model...")
        model = tf.keras.models.load_model(full_in, compile=False)

        print("  [2] Exporting as SavedModel (TensorFlow 2.15+ format)...")
        model.export(saved_dir)

        print("  [3] Initializing TFLite converter...")
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_dir)
        converter.experimental_new_converter = True
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter.target_spec.supported_types = [tf.float16]  # smaller FP16 weights

        # --- Optimization Toggle ---
        if ENABLE_OPTIMIZATION:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            print("  Optimization enabled: DEFAULT (smaller + faster model)")
        else:
            print("  Optimization disabled: full-precision float32 model")

        print("  [4] Converting... This may take several minutes.")
        tflite_model = converter.convert()
        print("  [5] Conversion completed successfully.")

        # Save the TFLite model
        with open(full_out, "wb") as f:
            f.write(tflite_model)

        tflite_size = len(tflite_model) / (1024 * 1024)
        print(f"Saved '{tflite_model_path}' ({tflite_size:.2f} MB)")

    except Exception as e:
        print(f"ERROR during conversion: {e}")

    finally:
        # --- Clean up temporary SavedModel folder ---
        if os.path.exists(saved_dir):
            try:
                shutil.rmtree(saved_dir)
                print(f"Cleaned up temporary folder: {tmp_folder_name}")
            except Exception as cleanup_err:
                print(f"Could not delete temp folder: {cleanup_err}")


# ======================================================
# MAIN EXECUTION
# ======================================================
def main():
    print("\n--- Starting TensorFlow Lite Conversion Process ---")
    CWD = os.getcwd()
    print(f"--- [DEBUG] Current working directory: {CWD} ---")

    for keras_file, tflite_file in MODELS_TO_CONVERT.items():
        convert_model_to_tflite(keras_file, tflite_file)

    print("\n--- All conversions complete. ---")


if __name__ == '__main__':
    main()
