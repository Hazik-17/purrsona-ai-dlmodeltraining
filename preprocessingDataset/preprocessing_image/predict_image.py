import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.models import load_model

# This small script loads one image and shows the model guess
# Input a model file and an image path
# Output printed class name and confidence score
MODEL_PATH = 'cat_breed_mobilenetv3_training_corrected.h5'
IMG_PATH = 'output\Bengal\Bengal_1.jpg'
IMG_SIZE = (224, 224)
CLASS_NAMES = ['Abyssinian', 'Bengal', 'Birman', 'British', 'Maine', 'Persian']

# Load the model
model = load_model(MODEL_PATH)

# Load the image and make it ready for the model
img = image.load_img(IMG_PATH, target_size=IMG_SIZE)
img_array = image.img_to_array(img)
img_array = preprocess_input(img_array)
img_array = np.expand_dims(img_array, axis=0)

# Run the model and show the best guess
pred_probs = model.predict(img_array)
pred_index = np.argmax(pred_probs[0])
confidence = pred_probs[0][pred_index] * 100

print(f"Predicted Class: {CLASS_NAMES[pred_index]}")
print(f"Confidence: {confidence:.2f}%")
