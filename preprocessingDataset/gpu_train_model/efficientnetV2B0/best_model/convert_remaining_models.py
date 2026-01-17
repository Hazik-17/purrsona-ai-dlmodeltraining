import os
# Force CPU to prevent GPU memory deadlocks (Crucial for success)
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
# 1. Gatekeeper Config (Cat vs Not Cat)
GATEKEEPER_IN = 'cat_vs_not_cat.keras'
GATEKEEPER_OUT = 'gatekeeper_model.tflite'
GATEKEEPER_CLASSES = 2  # [cat, not_cat]

# 2. Expert Config (The Specialized 5-Breed Model)
EXPERT_IN = 'similar_breed_expert_effnet_v1.keras'
EXPERT_OUT = 'similar_breed_expert_model.tflite'
EXPERT_CLASSES = 5 # Based on your previous comment

# Shared Architecture Settings (Must match training script)
IMG_SIZE = (224, 224)
DENSE_UNITS = 256
DROPOUT_RATE = 0.5

def transplant_and_convert(input_path, output_path, num_classes):
    print(f"\n==================================================")
    print(f" PROCESSING: {input_path}")
    print(f" TARGET CLASSES: {num_classes}")
    print(f"==================================================")

    if not os.path.exists(input_path):
        print(f"❌ Error: File '{input_path}' not found!")
        return

    # 1. Force Float32 Policy (Global)
    tf.keras.mixed_precision.set_global_policy('float32')

    try:
        # 2. Build a FRESH, CLEAN Model Architecture
        print("  [1] Building clean architecture...")
        
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
            # The output layer changes based on num_classes
            Dense(num_classes, activation='softmax', dtype='float32')
        ])
        
        # Initialize shapes with a dummy pass
        dummy_input = tf.ones((1, 224, 224, 3))
        _ = clean_model(dummy_input)

        # 3. Load the OLD model to grab weights
        print("  [2] Loading old model weights...")
        old_model = tf.keras.models.load_model(input_path, compile=False)
        
        # 4. TRANSPLANT WEIGHTS
        print("  [3] Transplanting weights...")
        try:
            clean_model.set_weights(old_model.get_weights())
        except ValueError as ve:
            print(f"\n❌ SHAPE MISMATCH ERROR for {input_path}")
            print(f"Script expected {num_classes} classes.")
            print("Please verify how many classes this model actually has.")
            raise ve

        # 5. Convert
        print("  [4] Initializing Converter...")
        converter = tf.lite.TFLiteConverter.from_keras_model(clean_model)
        
        # Pure TFLite Settings
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.target_spec.supported_types = [tf.float32]
        
        # Static Shape Helper
        def representative_dataset_gen():
            for _ in range(1):
                yield [np.zeros((1, 224, 224, 3), dtype=np.float32)]
        converter.representative_dataset = representative_dataset_gen

        print("  [5] Converting... (This may take 1-2 mins)")
        tflite_model = converter.convert()

        # 6. Save
        with open(output_path, "wb") as f:
            f.write(tflite_model)

        size_mb = len(tflite_model) / (1024 * 1024)
        print(f"✅ SUCCESS! Generated: {output_path} ({size_mb:.2f} MB)")

    except Exception as e:
        print(f"❌ FAILED: {e}")

if __name__ == '__main__':
    # Run Gatekeeper
    transplant_and_convert(GATEKEEPER_IN, GATEKEEPER_OUT, GATEKEEPER_CLASSES)
    
    # Run Expert
    transplant_and_convert(EXPERT_IN, EXPERT_OUT, EXPERT_CLASSES)