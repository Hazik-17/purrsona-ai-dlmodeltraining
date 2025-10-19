import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Path to your saved .keras model file
MODEL_PATH = 'preprocessingDataset\cat_breed_mobilenetv3_training_corrected.keras'#'cat_breed_mobilenetv3_training_corrected.keras'
IMG_SIZE = (224, 224)

# Load the saved Keras model
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")

# Map class indices to names (must match those used during training)
class_indices = {'Bengal': 0, 'Bombay': 1, 'Maine': 2, 'Persian': 3, 'Ragdoll': 4, 'Sphynx': 5}
idx_to_class = {v: k for k, v in class_indices.items()}

def load_and_preprocess_image(img_path):
    """Load and preprocess image for MobileNetV3 model."""
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # MobileNetV3 preprocessing
    return img_array, img

def predict_image(img_path):
    img_array, pil_img = load_and_preprocess_image(img_path)
    preds = model.predict(img_array)
    class_idx = np.argmax(preds[0])
    confidence = preds[0][class_idx]
    predicted_class = idx_to_class.get(class_idx, "Unknown")
    return predicted_class, confidence, pil_img

if __name__ == '__main__':
    # Replace this path by any test image path to try
    test_image_path = 'output_3breeds/test/Bengal/Bengal_68.jpg'

    predicted_class, confidence, pil_img = predict_image(test_image_path)

    print(f"Predicted Breed: {predicted_class} with confidence {confidence*100:.2f}%")

    # Optionally visualize the image with the prediction
    plt.imshow(pil_img)
    plt.title(f"Predicted: {predicted_class} ({confidence*100:.2f}%)")
    plt.axis('off')
    plt.show()
