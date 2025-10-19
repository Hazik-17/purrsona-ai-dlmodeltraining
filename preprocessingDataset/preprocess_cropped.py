import os
import shutil
import cv2
import numpy as np
import random
import csv
from sklearn.model_selection import train_test_split

# === Configuration ===
IMG_DIR = 'cropped_images'
OUTPUT_DIR = 'output_cropped'
SELECTED_BREEDS = ['Persian', 'Sphynx', 'Bombay', 'Bengal', 'Ragdoll', 'Maine']
TARGET_SIZE = (224, 224)
TEST_SIZE = 0.2
AUG_PER_IMAGE = 2

def get_breed_from_filename(filename):
    return filename.split('_')[0]

def augment_image(img):
    aug_list = []
    aug_list.append(cv2.flip(img, 1))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), random.randint(-15, 15), 1)
    aug_list.append(cv2.warpAffine(img, M, (w, h)))
    return aug_list

def preprocess_and_save(img_path, dest_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.resize(img, TARGET_SIZE)
    img = img / 255.0
    cv2.imwrite(dest_path, (img * 255).astype(np.uint8))
    return img

def main():
    breed_images = {b: [] for b in SELECTED_BREEDS}
    for file in os.listdir(IMG_DIR):
        if not file.endswith('.jpg'):
            continue
        breed = get_breed_from_filename(file)
        if breed in breed_images:
            breed_images[breed].append(file)

    label_map = {b: i for i, b in enumerate(SELECTED_BREEDS)}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_labels, test_labels = [], []

    for breed, files in breed_images.items():
        train_files, test_files = train_test_split(files, test_size=TEST_SIZE, random_state=42)
        for split, file_list in [('train', train_files), ('test', test_files)]:
            out_dir = os.path.join(OUTPUT_DIR, split, breed)
            os.makedirs(out_dir, exist_ok=True)
            for f in file_list:
                src = os.path.join(IMG_DIR, f)
                dst = os.path.join(out_dir, f)
                img = preprocess_and_save(src, dst)
                if img is not None:
                    (train_labels if split == 'train' else test_labels).append([f, breed, label_map[breed]])
                    if split == 'train':
                        for i, aug_img in enumerate(augment_image(img)):
                            aug_name = f.replace('.jpg', f'_aug{i+1}.jpg')
                            aug_dst = os.path.join(out_dir, aug_name)
                            cv2.imwrite(aug_dst, (aug_img * 255).astype(np.uint8))
                            train_labels.append([aug_name, breed, label_map[breed]])

    with open(os.path.join(OUTPUT_DIR, 'labels.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'breed', 'label'])
        writer.writerows(train_labels + test_labels)

    print("✅ Preprocessing complete. Output in 'output_cropped/'")

if __name__ == "__main__":
    main()
