import os
import cv2
import numpy as np

# Simple settings for input and output folders
IMAGES_DIR = 'images'  # folder with raw images
ANNOTATION_FILE = 'annotations/list.txt'
OUTPUT_DIR = 'output'  # where to save processed images
TARGET_SIZE = (224, 224)


# This function reads the annotation file and finds cat image names
# Input the annotation file path
# Output a map from breed to list of image file names
def get_cat_images(annotation_file):
    cat_images = {}
    with open(annotation_file, 'r') as f:
        lines = f.readlines()[6:]
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            image_name, class_id, species, breed_id = parts
            if image_name[0].isupper():
                breed = image_name.split('_')[0]
                if breed not in cat_images:
                    cat_images[breed] = []
                cat_images[breed].append(image_name + '.jpg')
    return cat_images


# This function resizes and saves images into breed folders
# Input a map of breed to image file names
# Output image files written to the output folder
def preprocess_images(cat_images_dict):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for breed, image_list in cat_images_dict.items():
        breed_dir = os.path.join(OUTPUT_DIR, breed)
        os.makedirs(breed_dir, exist_ok=True)

        for img_file in image_list:
            img_path = os.path.join(IMAGES_DIR, img_file)
            if not os.path.exists(img_path):
                print(f'[Skip] Not found {img_path}')
                continue

            img = cv2.imread(img_path)
            if img is None:
                print(f'[Warning] Failed to load {img_path}')
                continue

            img = cv2.resize(img, TARGET_SIZE)
            img = img / 255.0

            save_path = os.path.join(breed_dir, img_file)
            cv2.imwrite(save_path, (img * 255).astype(np.uint8))
            print(f'Saved {save_path}')


if __name__ == "__main__":
    print('Filtering cat images from annotation list')
    cat_images = get_cat_images(ANNOTATION_FILE)
    print(f'Detected {len(cat_images)} cat breeds')

    print('Preprocessing and saving images')
    preprocess_images(cat_images)
    print('Done')
