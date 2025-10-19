import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os

# === CONFIG ===
MODEL_PATH = 'cat_breed_mobilenetv3_training_corrected.keras'
IMG_SIZE = (224, 224)
CLASS_LABELS = [
    'Abyssinian',
    'Bengal',
    'Birman',
    'Bombay',
    'British',
    'Egyptian',
    'Maine',
    'Persian',
    'Ragdoll',
    'Russian',
    'Siamese',
    'Sphynx'
]

# === Load model ===
model = load_model(MODEL_PATH)
print("✅ Model loaded.")

# === Load and preprocess image ===
def load_and_prepare_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0), img

# === Predict and show top 3 breeds ===
def predict_breed(img_path):
    processed_img, display_img = load_and_prepare_image(img_path)
    prediction = model.predict(processed_img)[0]  # shape: (num_classes,)
    
    # Get top 3 predictions
    top_indices = prediction.argsort()[-3:][::-1]  # highest to lowest
    top_labels = [CLASS_LABELS[i] for i in top_indices]
    top_confidences = [prediction[i] * 100 for i in top_indices]

    # Display results
    print("\n📊 Top 3 Predicted Breeds:")
    for i in range(3):
        print(f"  {top_labels[i]}: {top_confidences[i]:.2f}%")

    # Display image with top prediction
    plt.imshow(display_img)
    plt.title(f"Top Prediction: {top_labels[0]} ({top_confidences[0]:.2f}%)")
    plt.axis('off')
    plt.show()

# === Main ===
if __name__ == '__main__':
    img_path = input("📷 Enter the path to your cat image: ")
    
    if os.path.exists(img_path):
        predict_breed(img_path)
    else:
        print("❌ File not found. Please check the path and try again.")
