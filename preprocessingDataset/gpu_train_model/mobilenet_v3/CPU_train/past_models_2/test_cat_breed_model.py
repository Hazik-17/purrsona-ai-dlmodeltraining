import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.preprocessing import image

# Config
IMG_SIZE = (224, 224)
BATCH_SIZE = 1  # for testing single images
NUM_CLASSES = 6  # your number of classes
WEIGHTS_PATH = 'cat_breed_mobilenetv3_weights.h5'  # path to your saved weights

# Rebuild Model Architecture Exactly As Training
base_model = MobileNetV3Small(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # or True if you want to further fine-tune

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

# Compile model (necessary before loading weights)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load weights
model.load_weights(WEIGHTS_PATH)
print("Model weights loaded.")

# Class index map (must match training)
class_indices = {'Bengal': 0, 'Bombay': 1, 'Maine': 2, 'Persian': 3, 'Ragdoll': 4, 'Sphynx': 5}
idx_to_class = {v: k for k, v in class_indices.items()}

def predict_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    class_idx = np.argmax(preds)
    confidence = preds[0][class_idx]
    predicted_class = idx_to_class[class_idx]
    return predicted_class, confidence

# Example usage: predict on a single image path
test_image_path = 'output_3breeds/test/Bengal/some_image.jpg'  # change to an actual image path
pred_class, conf = predict_image(test_image_path)
print(f"Prediction: {pred_class} with confidence {conf:.4f}")
