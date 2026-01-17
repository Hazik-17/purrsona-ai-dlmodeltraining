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
import csv

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_prediction_resources(model_path, class_indices_path):
    """Loads a trained model and its corresponding class indices map."""
    print("--- Loading Model Resources ---")
    try:
        print(f"--> Loading model from: {model_path}")
        # The .keras format is robust and handles custom objects automatically.
        model = load_model(model_path, compile=False)
        print("    ✅ Model loaded successfully.")
    except Exception as e:
        print(f"    ❌ Error loading model: {e}")
        sys.exit()

    try:
        with open(class_indices_path, 'r') as f:
            class_indices = json.load(f)
        class_labels = {v: k for k, v in class_indices.items()}
        print(f"--> Class labels loaded successfully: {list(class_labels.values())}")
    except Exception as e:
        print(f"    ❌ Error loading {class_indices_path}: {e}")
        sys.exit()

    return model, class_labels

def predict_image(model, image_path, img_size):
    """Loads, preprocesses, and predicts a single image."""
    try:
        img = load_img(image_path, target_size=img_size)
        img_array = img_to_array(img)
        img_batch = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(img_batch)
        
        # Returns the array of probabilities for the model's classes
        predictions = model.predict(preprocessed_img, verbose=0)[0]
        return predictions
    except Exception as e:
        print(f"    ❌ Error processing image {image_path}: {e}")
        return None

def display_prediction(image_path, top_predictions):
    """Displays the image with the top predictions overlaid."""
    img = plt.imread(image_path)
    plt.imshow(img)
    
    top_label, top_confidence = top_predictions[0]
    
    title_text = f"Prediction: {top_label} ({top_confidence:.2%})"
    
    other_preds_text = "\n".join([f"{label}: {confidence:.2%}" for label, confidence in top_predictions[1:]])
    
    plt.title(title_text, fontsize=14, color='white', backgroundcolor='black')
    plt.text(5, 25, other_preds_text, color='white', backgroundcolor='black', fontsize=10, verticalalignment='top')
    plt.axis('off')
    plt.show()

def process_input_path(model, class_labels, input_path, img_size, show_visuals, output_csv):
    """Processes either a single image or a directory of images."""
    if os.path.isdir(input_path):
        print(f"\n--> Processing all images in directory: {input_path}")
        report_data = []
        image_files = [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for filename in image_files:
            image_path = os.path.join(input_path, filename)
            predictions = predict_image(model, image_path, img_size)
            if predictions is not None:
                # Determine how many classes the model has for top-k predictions
                num_classes = len(predictions)
                k = min(3, num_classes) # Show top 3 or fewer if model has fewer classes
                
                top_k_indices = predictions.argsort()[-k:][::-1]
                top_k_preds = [(class_labels[i], predictions[i]) for i in top_k_indices]
                
                print(f"  - {filename}: {top_k_preds[0][0]} ({top_k_preds[0][1]:.2%})")
                report_data.append([filename, top_k_preds[0][0], f"{top_k_preds[0][1]:.2%}"])

        if output_csv:
            with open(output_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Filename', 'Predicted Breed', 'Confidence'])
                writer.writerows(report_data)
            print(f"\n--> ✅ Prediction report saved to {output_csv}")

    elif os.path.isfile(input_path):
        print(f"\n--> Processing single image: {input_path}")
        predictions = predict_image(model, input_path, img_size)
        if predictions is not None:
            num_classes = len(predictions)
            k = min(3, num_classes)
            
            top_k_indices = predictions.argsort()[-k:][::-1]
            top_predictions = [(class_labels[i], predictions[i]) for i in top_k_indices]
            
            print(f"\n=== Top {k} Predictions ===")
            for label, confidence in top_predictions:
                print(f"  - {label:<20} | Confidence: {confidence:.2%}")

            if show_visuals:
                display_prediction(input_path, top_predictions)
    else:
        print(f"    ❌ Error: Input path is not a valid file or directory: {input_path}")

def main():
    parser = argparse.ArgumentParser(description="Universal Cat Breed Model Tester")
    parser.add_argument("input_path", type=str, help="Path to a single image or a directory of images.")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained .keras model file to test.")
    parser.add_argument("--labels", type=str, required=True, help="Path to the corresponding class indices JSON file.")
    parser.add_argument("--img_size", type=int, default=224, help="Image size (width and height).")
    parser.add_argument("--no-visuals", action="store_true", help="Do not display the image with predictions.")
    parser.add_argument("--output_csv", type=str, default="test_report.csv", help="Filename for the CSV report when processing a directory.")
    
    args = parser.parse_args()

    img_dimensions = (args.img_size, args.img_size)
    model, class_labels = load_prediction_resources(args.model, args.labels)
    
    process_input_path(model, class_labels, args.input_path, img_dimensions, not args.no_visuals, args.output_csv)

if __name__ == "__main__":
    main()

