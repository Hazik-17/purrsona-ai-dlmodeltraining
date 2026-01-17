import os
# Force CPU to prevent GPU memory deadlocks
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.regularizers import l2

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_MODEL_PATH = 'breed_expert_efficientnetv2b0_v2.keras'
OUTPUT_TFLITE_PATH = 'generalist_breed_model.tflite'

# Must match your training script exactly
IMG_SIZE = (224, 224)
DENSE_UNITS = 256
DROPOUT_RATE = 0.5
NUM_CLASSES = 12 

def transplant_and_convert():
    print(f"--- Performing Weight Transplant for {INPUT_MODEL_PATH} ---")

    # 1. Force Float32 Policy (Global)
    tf.keras.mixed_precision.set_global_policy('float32')

    try:
        # 2. Build a FRESH, CLEAN Model Architecture
        print("  [1] Building fresh model architecture...")
        
        base_model = EfficientNetV2B0(
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
            include_top=False,
            weights='imagenet' 
        )
        base_model.trainable = False

        clean_model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(DENSE_UNITS, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(DROPOUT_RATE),
            Dense(NUM_CLASSES, activation='softmax', dtype='float32')
        ])
        
        # --- THE FIX: Initialize layers with a dummy pass instead of .build() ---
        print("  [1b] Initializing layers...")
        # We run one fake image through the model. This forces Keras to 
        # initialize the Dense/BatchNormal layers with the correct shapes.
        dummy_input = tf.ones((1, 224, 224, 3))
        _ = clean_model(dummy_input)

        # 3. Load the OLD model just to grab weights
        print("  [2] Loading old model to extract weights...")
        old_model = tf.keras.models.load_model(INPUT_MODEL_PATH, compile=False)
        
        # 4. TRANSPLANT WEIGHTS
        print("  [3] Transplanting weights...")
        try:
            clean_model.set_weights(old_model.get_weights())
        except ValueError as ve:
            print("\n!!! SHAPE MISMATCH !!!")
            print("Your trained model has a different structure than this script.")
            print("Check NUM_CLASSES or DENSE_UNITS settings.")
            raise ve

        # 5. Convert the CLEAN model
        print("  [4] Initializing Converter on clean model...")
        converter = tf.lite.TFLiteConverter.from_keras_model(clean_model)
        
        # PURE TFLITE SETTINGS
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.target_spec.supported_types = [tf.float32]
        
        # OPTIONAL: Define Static Input Shape for TFLite
        # This helps preventing the "FlexOps" auto-selection
        def representative_dataset_gen():
            for _ in range(1):
                yield [np.zeros((1, 224, 224, 3), dtype=np.float32)]
        converter.representative_dataset = representative_dataset_gen

        print("  [5] Converting... (Please wait)")
        tflite_model = converter.convert()

        # 6. Save
        with open(OUTPUT_TFLITE_PATH, "wb") as f:
            f.write(tflite_model)

        size_mb = len(tflite_model) / (1024 * 1024)
        print(f"\n✅ SUCCESS! Generated: {OUTPUT_TFLITE_PATH}")
        print(f"📊 Model Size: {size_mb:.2f} MB")
        print("🚀 Copy this file to your assets/models folder and rebuild Flutter.")

    except Exception as e:
        print(f"\n❌ TRANSPLANT FAILED: {e}")

if __name__ == '__main__':
    transplant_and_convert()