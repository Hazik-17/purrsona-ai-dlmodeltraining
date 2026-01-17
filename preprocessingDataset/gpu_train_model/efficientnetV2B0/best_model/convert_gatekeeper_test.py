import tensorflow as tf
import os

# Convert a saved Keras gatekeeper model using SavedModel -> TFLite flow.
keras_model_path = "cat_vs_not_cat.keras"
saved_model_dir = "tmp_gatekeeper_saved"
tflite_model_path = "gatekeeper_model.tflite"

print("Loading Keras model...")
model = tf.keras.models.load_model(keras_model_path, compile=False)

print("Exporting to SavedModel directory...")
model.export(saved_model_dir)

print("Converting SavedModel to TFLite (using Select TF ops)...")
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter.experimental_new_converter = True
converter.target_spec.supported_types = [tf.float16]

try:
    tflite_model = converter.convert()
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)
    print(f"Conversion successful. Saved at: {os.path.abspath(tflite_model_path)}")
except Exception as e:
    print(f"Conversion failed: {e}")
