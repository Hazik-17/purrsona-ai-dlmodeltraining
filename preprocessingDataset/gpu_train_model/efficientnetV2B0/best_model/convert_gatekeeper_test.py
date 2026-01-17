import tensorflow as tf
import os

keras_model_path = "cat_vs_not_cat.keras"
saved_model_dir = "tmp_gatekeeper_saved"
tflite_model_path = "gatekeeper_model.tflite"

print("[1] Loading Keras model...")
model = tf.keras.models.load_model(keras_model_path, compile=False)

print("[2] Exporting as SavedModel (TensorFlow 2.15+ uses model.export)...")
model.export(saved_model_dir)  # ✅ correct method for exporting SavedModel

print("[3] Starting TFLite conversion (with Select TF Ops)...")
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter.experimental_new_converter = True
# optional: comment this line if it fails
converter.target_spec.supported_types = [tf.float16]

try:
    tflite_model = converter.convert()
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)
    print(f"[4] ✅ Conversion successful! Saved as: {os.path.abspath(tflite_model_path)}")
except Exception as e:
    print(f"[4] ❌ Conversion failed: {e}")
