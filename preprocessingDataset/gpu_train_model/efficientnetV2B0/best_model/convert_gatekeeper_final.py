import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU

import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.regularizers import l2

# Simple conversion settings
INPUT_PATH = 'cat_vs_not_cat.keras'
OUTPUT_PATH = 'gatekeeper_model.tflite'
IMG_SIZE = (224, 224)
DENSE_UNITS = 256
DROPOUT_RATE = 0.5

def build_architecture(num_classes, activation_type):
    """Make a clean model with given output units and activation.

    Returns: a Keras model ready for weights to be set.
    """
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
    print(f"Processing Gatekeeper: {INPUT_PATH}")
    tf.keras.mixed_precision.set_global_policy('float32')

    if not os.path.exists(INPUT_PATH):
        print("Error: Input file not found.")
        return

    print("Loading old model weights...")
    old_model = tf.keras.models.load_model(INPUT_PATH, compile=False)
    weights = old_model.get_weights()

    # Try two possible output shapes: 2 units (softmax) or 1 unit (sigmoid)
    print("Trying 2-output softmax architecture...")
    try:
        clean_model = build_architecture(2, 'softmax')
        clean_model.set_weights(weights)
        print("Matched 2-output model.")
    except ValueError:
        print("2-output failed, trying 1-output sigmoid...")
        try:
            clean_model = build_architecture(1, 'sigmoid')
            clean_model.set_weights(weights)
            print("Matched 1-output model.")
        except ValueError as e:
            print("Fatal: could not match model architecture.")
            print(f"Details: {e}")
            return

    # Convert the matched model to TFLite
    print("Converting to TFLite...")
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

    print(f"Saved TFLite file: {OUTPUT_PATH}")
    print(f"Size (MB): {len(tflite_model)/(1024*1024):.2f}")

if __name__ == '__main__':
    try_conversion()