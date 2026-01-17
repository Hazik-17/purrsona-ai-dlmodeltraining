import os
import shutil
import tensorflow as tf

ENABLE_OPTIMIZATION = False

MODELS_TO_CONVERT = {
    'cat_vs_not_cat.keras': 'gatekeeper_model.tflite',
    'breed_expert_efficientnetv2b0_v2.keras': 'generalist_breed_model.tflite',
    'similar_breed_expert_effnet_v1.keras': 'similar_breed_expert_model.tflite'
}


def convert_model_to_tflite(keras_model_path, tflite_model_path):
    """Load a Keras model, convert to TFLite, and save the file."""
    print(f"Converting: {keras_model_path}")

    full_in = os.path.abspath(keras_model_path)
    full_out = os.path.join(os.path.dirname(__file__), tflite_model_path)
    tmp_folder_name = f"tmp_saved_{os.path.splitext(keras_model_path)[0]}"
    saved_dir = os.path.join(os.path.dirname(__file__), tmp_folder_name)

    print(f"  Input: {full_in}")
    print(f"  Output: {full_out}")

    if not os.path.exists(full_in):
        print("File not found.")
        return

    try:
        model = tf.keras.models.load_model(full_in, compile=False)
        model.export(saved_dir)

        converter = tf.lite.TFLiteConverter.from_saved_model(saved_dir)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

        if ENABLE_OPTIMIZATION:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            print("  Optimization enabled")
        else:
            print("  Optimization disabled")

        print("  Converting...")
        tflite_model = converter.convert()

        with open(full_out, "wb") as f:
            f.write(tflite_model)

        tflite_size = len(tflite_model) / (1024 * 1024)
        print(f"Saved: {tflite_model_path} ({tflite_size:.2f} MB)")

    except Exception as e:
        print(f"Conversion error: {e}")

    finally:
        if os.path.exists(saved_dir):
            try:
                shutil.rmtree(saved_dir)
                print(f"Removed temp folder: {tmp_folder_name}")
            except Exception as cleanup_err:
                print(f"Could not delete temp folder: {cleanup_err}")


def main():
    print("Starting TFLite conversions")
    for keras_file, tflite_file in MODELS_TO_CONVERT.items():
        convert_model_to_tflite(keras_file, tflite_file)
    print("All conversions finished")


if __name__ == '__main__':
    main()
