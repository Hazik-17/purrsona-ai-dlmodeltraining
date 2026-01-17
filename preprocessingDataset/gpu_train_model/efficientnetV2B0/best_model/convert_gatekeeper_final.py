import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Force CPU

import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.regularizers import l2

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_PATH = 'cat_vs_not_cat.keras'
OUTPUT_PATH = 'gatekeeper_model.tflite'
IMG_SIZE = (224, 224)
DENSE_UNITS = 256
DROPOUT_RATE = 0.5

def build_architecture(num_classes, activation_type):
    """Builds the clean Float32 architecture."""
    base_model = EfficientNetV2B0(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(DENSE_UNITS, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(DROPOUT_RATE),
        Dense(num_classes, activation=activation_type, dtype='float32')
    ])
    
    # Dummy pass to init shapes
    _ = model(tf.ones((1, 224, 224, 3)))
    return model

def try_conversion():
    print(f"--- Processing Gatekeeper: {INPUT_PATH} ---")
    tf.keras.mixed_precision.set_global_policy('float32')

    if not os.path.exists(INPUT_PATH):
        print("❌ Error: Input file not found.")
        return

    print("  [1] Loading old model weights...")
    old_model = tf.keras.models.load_model(INPUT_PATH, compile=False)
    weights = old_model.get_weights()

    # --- ATTEMPT 1: Try 2 Output Neurons (Softmax) ---
    print("  [2] Attempting Strategy A: 2 Classes (Softmax)...")
    try:
        clean_model = build_architecture(2, 'softmax')
        clean_model.set_weights(weights)
        print("     -> Success! Structure matches 2 neurons.")
    except ValueError:
        print("     -> Failed. Weights don't match 2 neurons.")
        
        # --- ATTEMPT 2: Try 1 Output Neuron (Sigmoid) ---
        print("  [2] Attempting Strategy B: 1 Class (Sigmoid/Binary)...")
        try:
            clean_model = build_architecture(1, 'sigmoid')
            clean_model.set_weights(weights)
            print("     -> Success! Structure matches 1 neuron.")
        except ValueError as e:
            print(f"\n❌ FATAL ERROR: Could not match architecture.")
            print(f"Details: {e}")
            return

    # --- CONVERSION ---
    print("  [3] Converting to Pure TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(clean_model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.target_spec.supported_types = [tf.float32]
    
    def representative_dataset_gen():
        for _ in range(1):
            yield [np.zeros((1, 224, 224, 3), dtype=np.float32)]
    converter.representative_dataset = representative_dataset_gen

    tflite_model = converter.convert()

    with open(OUTPUT_PATH, "wb") as f:
        f.write(tflite_model)

    print(f"\n✅ SUCCESS! Generated: {OUTPUT_PATH}")
    print(f"📊 Size: {len(tflite_model)/(1024*1024):.2f} MB")

if __name__ == '__main__':
    try_conversion()