import os
import shutil
import glob
import random

# Configuration
# Folder with 90 other animal folders (not cats)
NOT_CAT_SOURCE_DIR = r'../others_animal'

# Folder with your 12-breed cat dataset (train/test)
CAT_SOURCE_DIR = '../output'

# Where to save the new binary dataset
DEST_DIR = 'cat_vs_not_cat_dataset'

# Split ratio and randomness for train/test split
TEST_SPLIT_RATIO = 0.2
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def process_cat_images(cat_source_dir, dest_dir):
    """Copy cat images from 12-breed dataset to new 'cat' folders.

    Inputs: source folder and destination base folder.
    Output: counts of copied train and test images.
    """
    print(f"\n--- Processing 'cat' images from '{cat_source_dir}' ---")
    cat_train_count = 0
    cat_test_count = 0
    
    # Create destination folders
    dest_train_cat = os.path.join(dest_dir, 'train', 'cat')
    dest_test_cat = os.path.join(dest_dir, 'test', 'cat')
    os.makedirs(dest_train_cat, exist_ok=True)
    os.makedirs(dest_test_cat, exist_ok=True)
    
    # 1. Process the 'train' split from your 12-breed dataset
    source_train_dir = os.path.join(cat_source_dir, 'train')
    if os.path.isdir(source_train_dir):
        # Find all images in all 12 subfolders
        train_images = glob.glob(os.path.join(source_train_dir, '*', '*.jpg'))
        train_images.extend(glob.glob(os.path.join(source_train_dir, '*', '*.png')))
        for img_path in train_images:
            shutil.copy(img_path, dest_train_cat)
            cat_train_count += 1
    
    # 2. Process the 'test' split from your 12-breed dataset
    source_test_dir = os.path.join(cat_source_dir, 'test')
    if os.path.isdir(source_test_dir):
        # Find all images in all 12 subfolders
        test_images = glob.glob(os.path.join(source_test_dir, '*', '*.jpg'))
        test_images.extend(glob.glob(os.path.join(source_test_dir, '*', '*.png')))
        for img_path in test_images:
            shutil.copy(img_path, dest_test_cat)
            cat_test_count += 1

    print(f"  - Copied {cat_train_count} 'cat' images to train set.")
    print(f"  - Copied {cat_test_count} 'cat' images to test set.")
    return cat_train_count, cat_test_count

def process_not_cat_images(not_cat_source_dir, dest_dir, split_ratio, seed):
    """Copy non-cat images from many animal folders into 'not_cat'.

    The function shuffles each folder, splits into train/test,
    and copies files to the destination.
    """
    print(f"\n--- Processing 'not_cat' images from '{not_cat_source_dir}' ---")
    not_cat_train_count = 0
    not_cat_test_count = 0

    # Create destination folders
    dest_train_not_cat = os.path.join(dest_dir, 'train', 'not_cat')
    dest_test_not_cat = os.path.join(dest_dir, 'test', 'not_cat')
    os.makedirs(dest_train_not_cat, exist_ok=True)
    os.makedirs(dest_test_not_cat, exist_ok=True)

    if not os.path.isdir(not_cat_source_dir):
        print(f"❌ ERROR: 'not_cat' source directory not found at '{not_cat_source_dir}'.")
        print("   Please edit the 'NOT_CAT_SOURCE_DIR' variable in this script.")
        return 0, 0

    animal_folders = [f for f in os.listdir(not_cat_source_dir) if os.path.isdir(os.path.join(not_cat_source_dir, f))]
    
    for animal_folder in animal_folders:
        # --- THIS IS THE KEY ---
        if animal_folder == 'cat':
            print(f"  - Skipping 'cat' folder in source.")
            continue # Skip the 'cat' folder

        # Find all images for this animal
        animal_path = os.path.join(not_cat_source_dir, animal_folder)
        images = glob.glob(os.path.join(animal_path, '*.jpg'))
        images.extend(glob.glob(os.path.join(animal_path, '*.png')))
        
        if not images:
            continue

        # Shuffle and split the images 80/20
        random.shuffle(images)
        split_index = int(len(images) * (1 - split_ratio))
        train_files = images[:split_index]
        test_files = images[split_index:]

        # Copy to the new dataset
        for f in train_files:
            shutil.copy(f, dest_train_not_cat)
            not_cat_train_count += 1
        for f in test_files:
            shutil.copy(f, dest_test_not_cat)
            not_cat_test_count += 1

    print(f"  - Copied {not_cat_train_count} 'not_cat' images to train set (from 89 folders).")
    print(f"  - Copied {not_cat_test_count} 'not_cat' images to test set (from 89 folders).")
    return not_cat_train_count, not_cat_test_count

def main():
    print("--- Starting Binary Dataset Creation ---")

    # Remove old output if present so we start fresh
    if os.path.exists(DEST_DIR):
        print(f"Removing old dataset directory: '{DEST_DIR}'")
        shutil.rmtree(DEST_DIR)

    # Copy cat images from the 12-breed dataset
    cat_train_count, cat_test_count = process_cat_images(CAT_SOURCE_DIR, DEST_DIR)

    # Copy many non-cat animals into the not_cat class
    not_cat_train_count, not_cat_test_count = process_not_cat_images(NOT_CAT_SOURCE_DIR, DEST_DIR, TEST_SPLIT_RATIO, RANDOM_SEED)

    print("\n--- Done: Dataset Creation Complete ---")
    print("Summary:")
    print(f"  Total Training Images: {cat_train_count + not_cat_train_count}")
    print(f"    - cat:     {cat_train_count}")
    print(f"    - not_cat: {not_cat_train_count}")
    print(f"  Total Test Images: {cat_test_count + not_cat_test_count}")
    print(f"    - cat:     {cat_test_count}")
    print(f"    - not_cat: {not_cat_test_count}")
    print(f"Dataset is ready in '{DEST_DIR}'")

if __name__ == "__main__":
    main()


"""

1.  **Save this Script:** Save the code above as `prepare_binary_dataset.py` in your main project folder.
2.  **CRITICAL:** Open the script and **edit the `NOT_CAT_SOURCE_DIR` variable** (line 12) to point to the folder on your computer that contains the 90 animal subfolders.
3.  **Run the Prep Script:**
    ```bash
    python prepare_binary_dataset.py
    ```
    This will take a minute and will create the new, large `cat_vs_not_cat_dataset` folder.
4.  **Run the Training Script:**
    Now, run your `train_gatekeeper_model_effnet.py` script (the one from before that uses `EfficientNetV2B0`). It's already configured to use the `cat_vs_not_cat_dataset` folder.
    ```bash
    python train_gatekeeper_model_effnet.py

"""