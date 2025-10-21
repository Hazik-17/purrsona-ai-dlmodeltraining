import os
import cv2
import numpy as np
import random
import csv
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# === Configuration ===
IMAGES_DIR = 'images'
ANNOTATION_FILE = 'annotations/list.txt'
OUTPUT_DIR = 'output_3breeds'
TARGET_SIZE = (224, 224)
TEST_SIZE = 0.2
AUGMENTATIONS_PER_IMAGE = 2
SELECTED_BREEDS = ['Persian', 'Sphynx', 'Bombay', 'Bengal', 'Ragdoll', 'Maine']

# === Utilities ===
def load_selected_cat_images(annotation_file, selected_breeds):
    cat_images = {}
    with open(annotation_file, 'r') as f:
        lines = f.readlines()[6:]
        for line in lines:
            parts = line.strip().split()
            img_name = parts[0] + '.jpg'
            if img_name[0].isupper():
                breed = img_name.split('_')[0]
                if breed in selected_breeds:
                    if breed not in cat_images:
                        cat_images[breed] = []
                    cat_images[breed].append(img_name)
    return cat_images

def augment_image(img):
    augmented = []
    augmented.append(cv2.flip(img, 1))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), random.randint(-15, 15), 1)
    augmented.append(cv2.warpAffine(img, M, (w, h)))
    return augmented

def preprocess_and_save(img_path, dest_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.resize(img, TARGET_SIZE)
    img = img / 255.0
    cv2.imwrite(dest_path, (img * 255).astype(np.uint8))
    return img

def main():
    cat_images = load_selected_cat_images(ANNOTATION_FILE, SELECTED_BREEDS)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_labels = []
    test_labels = []

    # Use ImageDataGenerator to get class indices
    breed_to_idx = {breed: i for i, breed in enumerate(SELECTED_BREEDS)}



    for breed, images in cat_images.items():
        train_imgs, test_imgs = train_test_split(images, test_size=TEST_SIZE, random_state=42)

        for split, split_imgs in [('train', train_imgs), ('test', test_imgs)]:
            breed_dir = os.path.join(OUTPUT_DIR, split, breed)
            os.makedirs(breed_dir, exist_ok=True)

            for img_file in split_imgs:
                src = os.path.join(IMAGES_DIR, img_file)
                dest = os.path.join(breed_dir, img_file)
                img = preprocess_and_save(src, dest)

                if img is not None:
                    (train_labels if split == 'train' else test_labels).append([img_file, breed, breed_to_idx[breed]])

                    if split == 'train':
                        aug_imgs = augment_image(img)
                        for i, aug in enumerate(aug_imgs):
                            aug_name = img_file.replace('.jpg', f'_aug{i+1}.jpg')
                            aug_dest = os.path.join(breed_dir, aug_name)
                            cv2.imwrite(aug_dest, (aug * 255).astype(np.uint8))
                            train_labels.append([aug_name, breed, breed_to_idx[breed]])

    # Save new labels.csv
    with open(os.path.join(OUTPUT_DIR, 'labels.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'breed', 'label'])
        writer.writerows(train_labels + test_labels)

    print("✅ Preprocessing done for 3 breeds!")

if __name__ == "__main__":
    main()
