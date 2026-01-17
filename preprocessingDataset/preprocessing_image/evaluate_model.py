import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os

# This script loads a saved model and shows the top breed guesses
# It shows the image and prints the top three breed names and scores

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

# Load the model file and print a short message
model = load_model(MODEL_PATH)
print('Model loaded')


# This function loads one image and makes it ready for the model
# Input a path to an image file
# Output the image array for the model and the image for display
def load_and_prepare_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0), img


# This function predicts the breed and prints the top three guesses
# Input a path to an image file
# Output prints top three breed names and shows the image
def predict_breed(img_path):
    processed_img, display_img = load_and_prepare_image(img_path)
    prediction = model.predict(processed_img)[0]

    # Find top three results from the model output
    top_indices = prediction.argsort()[-3:][::-1]
    top_labels = [CLASS_LABELS[i] for i in top_indices]
    top_confidences = [prediction[i] * 100 for i in top_indices]

    print('\nTop 3 Predicted Breeds:')
    for i in range(3):
        print(f'  {top_labels[i]}: {top_confidences[i]:.2f}%')

    # Show the image and the top result
    plt.imshow(display_img)
    plt.title(f'Top Prediction: {top_labels[0]} {top_confidences[0]:.2f}%')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    img_path = input('Enter the path to your cat image: ')
    if os.path.exists(img_path):
        predict_breed(img_path)
    else:
        print('File not found please check the path')
