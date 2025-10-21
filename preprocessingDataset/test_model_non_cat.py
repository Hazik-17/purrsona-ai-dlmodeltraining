import os
import sys
import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import matplotlib.pyplot as plt

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_binary_model(model_path, class_indices_path):
    """Loads the binary classifier model and its class indices."""
    try:
        print(f"--> Loading gatekeeper model from: {model_path}")
        model = load_model(model_path, compile=False)
        print("    ✅ Model loaded successfully.")
    except Exception as e:
        print(f"    ❌ Error loading model: {e}")
        sys.exit()

    try:
        with open(class_indices_path, 'r') as f:
            class_indices = json.load(f)
        print(f"--> Class labels loaded: {class_indices}")
    except Exception as e:
        print(f"    ❌ Error loading {class_indices_path}: {e}")
        sys.exit()
        
    return model, class_indices

def predict_binary(model, image_path, img_size):
    """Predicts if an image is a 'cat' or 'not_cat'."""
    try:
        img = load_img(image_path, target_size=img_size)
        img_array = img_to_array(img)
        img_batch = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(img_batch)
        
        # The model outputs a single value between 0 and 1
        prediction = model.predict(preprocessed_img, verbose=0)[0][0]
        return prediction
    except Exception as e:
        print(f"    ❌ Error processing image {image_path}: {e}")
        return None

def display_binary_prediction(image_path, label, confidence):
    """Displays the image with the binary prediction overlaid."""
    img = plt.imread(image_path)
    plt.imshow(img)
    
    title_text = f"Prediction: {label} ({confidence:.2%})"
    
    plt.title(title_text, fontsize=16, color='white', backgroundcolor='black')
    plt.axis('off')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Gatekeeper Model Tester (Cat vs. Not-Cat)")
    parser.add_argument("image_path", type=str, help="Path to a single image to test.")
    parser.add_argument("--model", type=str, default="gatekeeper_cat_vs_not_cat.keras", help="Path to the trained gatekeeper .keras model.")
    parser.add_argument("--labels", type=str, default="gatekeeper_class_indices.json", help="Path to the gatekeeper class indices JSON file.")
    parser.add_argument("--img_size", type=int, default=224, help="Image size (width and height).")
    parser.add_argument("--no-visuals", action="store_true", help="Do not display the image with the prediction.")
    
    args = parser.parse_args()

    img_dimensions = (args.img_size, args.img_size)
    model, class_indices = load_binary_model(args.model, args.labels)
    
    # In a binary model, class_indices tells us which label is 0 and which is 1
    # Example: {'cat': 0, 'not_cat': 1}
    # We want to find which label corresponds to the '0' index.
    cat_label_index = class_indices.get('cat', 0) 

    prediction_prob = predict_binary(model, args.image_path, img_dimensions)
    
    if prediction_prob is not None:
        # The model's output is a probability. Let's say 'cat' is class 0.
        # A low probability (e.g., < 0.5) means it's likely class 0 ('cat').
        # A high probability (e.g., > 0.5) means it's likely class 1 ('not_cat').
        
        if round(prediction_prob) == cat_label_index:
            # If the rounded prediction matches the 'cat' index, it's a cat.
            # Confidence is 1 minus the probability.
            predicted_label = "Cat"
            confidence = 1 - prediction_prob
        else:
            # Otherwise, it's not a cat.
            # Confidence is the raw probability.
            predicted_label = "Not Cat"
            confidence = prediction_prob

        print("\n=== Gatekeeper Result ===")
        print(f"  Prediction: {predicted_label}")
        print(f"  Confidence: {confidence:.2%}")

        if not args.no_visuals:
            display_binary_prediction(args.image_path, predicted_label, confidence)

if __name__ == "__main__":
    main()
