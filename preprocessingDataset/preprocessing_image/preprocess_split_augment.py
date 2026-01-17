import os
import cv2
import numpy as np
import random
import csv
from sklearn.model_selection import train_test_split

IMAGES_DIR = 'images'
ANNOTATION_FILE = 'annotations/list.txt'
OUTPUT_DIR = 'output'
TARGET_SIZE = (224, 224)
TEST_SIZE = 0.2
AUGMENTATIONS_PER_IMAGE = 2


# This function reads the annotation file and finds cat images
# Input the annotation file path
# Output a map of breed to list of image names
def load_cat_images(annotation_file):
    cat_images = {}
    with open(annotation_file, 'r') as f:
        lines = f.readlines()[6:]
        for line in lines:
            parts = line.strip().split()
            image_name = parts[0] + '.jpg'
            if image_name[0].isupper():
                breed = image_name.split('_')[0]
                if breed not in cat_images:
                    cat_images[breed] = []
                cat_images[breed].append(image_name)
    return cat_images


# This function makes new images by flipping or rotating
# Input an image array
# Output a small list of new images
def augment_image(img):
    augmented = []
    augmented.append(cv2.flip(img, 1))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle=random.randint(-15, 15), scale=1)
    augmented.append(cv2.warpAffine(img, M, (w, h)))
    return augmented[:AUGMENTATIONS_PER_IMAGE]


# This function loads saves and resizes an image
# Input path to source and path to destination
# Output True when save succeeded False if load failed
def preprocess_and_save(img_path, dest_path):
    img = cv2.imread(img_path)
    if img is None:
        return False
    img = cv2.resize(img, TARGET_SIZE)
    img = img / 255.0
    cv2.imwrite(dest_path, (img * 255).astype(np.uint8))
    return True


# Main run function that creates train test folders and labels
def main():
    print('Loading annotation file')
    cat_images = load_cat_images(ANNOTATION_FILE)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_labels = []
    test_labels = []

    breed_to_idx = {breed: idx for idx, breed in enumerate(sorted(cat_images.keys()))}

    for breed, images in cat_images.items():
        train_imgs, test_imgs = train_test_split(images, test_size=TEST_SIZE, random_state=42)

        for split_name, split_imgs in [('train', train_imgs), ('test', test_imgs)]:
            split_dir = os.path.join(OUTPUT_DIR, split_name, breed)
            os.makedirs(split_dir, exist_ok=True)

            for img_file in split_imgs:
                src = os.path.join(IMAGES_DIR, img_file)
                dest = os.path.join(split_dir, img_file)

                if preprocess_and_save(src, dest):
                    (train_labels if split_name == 'train' else test_labels).append([img_file, breed, breed_to_idx[breed]])

                    if split_name == 'train':
                        img = cv2.imread(dest)
                        aug_imgs = augment_image(img)
                        for i, aug in enumerate(aug_imgs):
                            aug_name = img_file.replace('.jpg', f'_aug{i+1}.jpg')
                            aug_dest = os.path.join(split_dir, aug_name)
                            cv2.imwrite(aug_dest, aug)
                            train_labels.append([aug_name, breed, breed_to_idx[breed]])

    with open(os.path.join(OUTPUT_DIR, 'labels.csv'), mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'breed', 'label'])
        writer.writerows(train_labels + test_labels)

    print(f'Done. Train {len(train_labels)} Test {len(test_labels)}')


if __name__ == "__main__":
    main()
