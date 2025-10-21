import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import json

# === Configuration ===
MODEL_PATH = 'preprocessingDataset\cat_breed_mobilenetv3_training_corrected.h5'
CLASS_INDICES_PATH = 'preprocessingDataset\class_indices.json'
IMG_SIZE = (224, 224)

# === Load Model ===
try:
    print(f"Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please make sure the model file is in the same directory.")
    sys.exit()

# === Load Class Indices ===
try:
    with open(CLASS_INDICES_PATH, 'r') as f:
        class_indices = json.load(f)
    
    # Create a reverse mapping from index to class name
    class_labels = {v: k for k, v in class_indices.items()}
    print(f"✅ Class labels loaded successfully: {class_labels}")

except Exception as e:
    print(f"Error loading {CLASS_INDICES_PATH}: {e}")
    print("Please make sure you have run the training script to create this file.")
    sys.exit()

def predict_image(image_path):
    """
    Loads, preprocesses, and predicts a single image.
    """
    try:
        # 1. Load Image
        # target_size must match the model's input shape
        img = load_img(image_path, target_size=IMG_SIZE)

        # 2. Convert to Array
        img_array = img_to_array(img)

        # 3. Expand Dimensions to create a "batch" of 1
        # Shape changes from (224, 224, 3) to (1, 224, 224, 3)
        img_batch = np.expand_dims(img_array, axis=0)

        # 4. Preprocess the Image
        # This uses the *exact* same preprocessing as the training script
        preprocessed_img = preprocess_input(img_batch)

        # 5. Make Prediction
        predictions = model.predict(preprocessed_img)

        # 6. Interpret Results
        # predictions[0] is the array of probabilities for all classes
        confidence = np.max(predictions[0])
        pred_index = np.argmax(predictions[0])
        
        # 7. Map index to class name
        pred_class_name = class_labels[pred_index]

        return pred_class_name, confidence

    except FileNotFoundError:
        return f"Error: Image file not found at {image_path}", 0
    except Exception as e:
        return f"Error processing image: {e}", 0

# === Main execution ===
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage: python test_model.py <path_to_your_image.jpg>")
    else:
        image_path = sys.argv[1]
        
        if not os.path.exists(image_path):
            print(f"Error: No file found at '{image_path}'")
        else:
            print(f"\n🧠 Analyzing image: {image_path}...")
            
            class_name, confidence = predict_image(image_path)
            
            if confidence > 0:
                print("\n=== Prediction Result ===")
                print(f"Predicted Breed:  {class_name}")
                print(f"Confidence:       {confidence * 100:.2f}%")