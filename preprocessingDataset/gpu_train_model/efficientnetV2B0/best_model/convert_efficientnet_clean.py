import os
# Force CPU to prevent GPU memory deadlocks
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.regularizers import l2

# Paths and simple settings
INPUT_MODEL_PATH = 'breed_expert_efficientnetv2b0_v2.keras'
OUTPUT_TFLITE_PATH = 'generalist_breed_model.tflite'

# Must match training script
IMG_SIZE = (224, 224)
DENSE_UNITS = 256
DROPOUT_RATE = 0.5
NUM_CLASSES = 12

def transplant_and_convert():
    """Build a clean model, copy weights from trained file, and export TFLite."""

    print(f"Starting transplant for {INPUT_MODEL_PATH}")
    tf.keras.mixed_precision.set_global_policy('float32')

    try:
        # Build a fresh model with the same shape as training
        print("Building model architecture...")
        base_model = EfficientNetV2B0(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), include_top=False, weights='imagenet')
        base_model.trainable = False

        clean_model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(DENSE_UNITS, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(DROPOUT_RATE),
            Dense(NUM_CLASSES, activation='softmax', dtype='float32')
        ])

        # Initialize layers by running one fake input through the model
        dummy_input = tf.ones((1, 224, 224, 3))
        _ = clean_model(dummy_input)

        # Load trained model to get weights
        print("Loading trained model weights...")
        old_model = tf.keras.models.load_model(INPUT_MODEL_PATH, compile=False)

        # Try to copy weights into clean model
        print("Transplanting weights...")
        try:
            clean_model.set_weights(old_model.get_weights())
        except ValueError as ve:
            print("Shape mismatch: check NUM_CLASSES or DENSE_UNITS.")
            raise ve

        # Convert the model to TFLite
        print("Converting to TFLite...")
        converter = tf.lite.TFLiteConverter.from_keras_model(clean_model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.target_spec.supported_types = [tf.float32]

        def representative_dataset_gen():
            for _ in range(1):
                yield [np.zeros((1, 224, 224, 3), dtype=np.float32)]
        converter.representative_dataset = representative_dataset_gen

        tflite_model = converter.convert()

        with open(OUTPUT_TFLITE_PATH, "wb") as f:
            f.write(tflite_model)

        size_mb = len(tflite_model) / (1024 * 1024)
        print(f"Saved: {OUTPUT_TFLITE_PATH} ({size_mb:.2f} MB)")

    except Exception as e:
        print(f"Transplant failed: {e}")

if __name__ == '__main__':
    transplant_and_convert()