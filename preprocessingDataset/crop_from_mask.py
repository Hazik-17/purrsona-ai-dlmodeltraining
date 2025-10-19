import os
import cv2
import numpy as np

IMG_DIR = 'images'
MASK_DIR = 'annotations/trimaps'
OUT_DIR = 'cropped_images'
os.makedirs(OUT_DIR, exist_ok=True)

def crop_cat_from_mask(img_path, mask_path, save_path):
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        print(f"⚠️ Skipping (not found): {img_path}")
        return

    # Create binary mask where label==1 (cat)
    binary = (mask == 1).astype(np.uint8)

    # Find contours on binary mask
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"⚠️ No cat found in mask: {img_path}")
        return

    # Find largest contour
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # Add margin (optional)
    margin = 10
    x, y = max(x - margin, 0), max(y - margin, 0)
    w, h = min(w + 2 * margin, img.shape[1] - x), min(h + 2 * margin, img.shape[0] - y)

    # Crop and save
    cropped = img[y:y+h, x:x+w]
    save_path_full = os.path.join(save_path, os.path.basename(img_path))
    cv2.imwrite(save_path_full, cropped)

# Process all cat images
for file in os.listdir(IMG_DIR):
    if not file.lower().endswith('.jpg'):
        continue
    if not file[0].isupper():
        continue  # skip dog images

    img_path = os.path.join(IMG_DIR, file)
    mask_path = os.path.join(MASK_DIR, file.replace('.jpg', '.png'))

    crop_cat_from_mask(img_path, mask_path, OUT_DIR)

print("✅ Cropping complete. Saved to 'cropped_images/'")
