import os
import shutil
import random
import glob

def prepare_binary_dataset(source_dir, dest_dir, train_split=0.8):
    # This function makes a cat versus not cat dataset from many animal folders
    # Input a folder with animal subfolders and a destination folder
    # Output a new folder with train and test subfolders for cat and not cat

    train_path = os.path.join(dest_dir, 'train')
    test_path = os.path.join(dest_dir, 'test')

    train_cat_path = os.path.join(train_path, 'cat')
    train_not_cat_path = os.path.join(train_path, 'not_cat')
    test_cat_path = os.path.join(test_path, 'cat')
    test_not_cat_path = os.path.join(test_path, 'not_cat')

    for path in [train_cat_path, train_not_cat_path, test_cat_path, test_not_cat_path]:
        os.makedirs(path, exist_ok=True)

    print(f'Created directory structure in {dest_dir}')

    # Copy cat images into train and test folders
    cat_source_path = os.path.join(source_dir, 'cat')
    cat_images = glob.glob(os.path.join(cat_source_path, '*.jpg'))
    random.shuffle(cat_images)

    split_index = int(len(cat_images) * train_split)
    train_cats = cat_images[:split_index]
    test_cats = cat_images[split_index:]

    for img_path in train_cats:
        shutil.copy(img_path, os.path.join(train_cat_path, os.path.basename(img_path)))
    for img_path in test_cats:
        shutil.copy(img_path, os.path.join(test_cat_path, os.path.basename(img_path)))

    print(f"Processed cat class {len(train_cats)} training {len(test_cats)} testing images")

    # Gather all not cat images from other folders
    not_cat_images = []
    animal_folders = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    for animal in animal_folders:
        if animal.lower() != 'cat':
            animal_path = os.path.join(source_dir, animal)
            images = glob.glob(os.path.join(animal_path, '*.jpg'))
            not_cat_images.extend(images)

    print(f'Found {len(not_cat_images)} not cat images from {len(animal_folders) - 1} classes')

    # Make a balanced set of not cat images to match cats
    random.shuffle(not_cat_images)
    balanced_not_cats = not_cat_images[:len(cat_images)]
    print(f'Balancing dataset by sampling {len(balanced_not_cats)} not cat images')

    split_index = int(len(balanced_not_cats) * train_split)
    train_not_cats = balanced_not_cats[:split_index]
    test_not_cats = balanced_not_cats[split_index:]

    for img_path in train_not_cats:
        shutil.copy(img_path, os.path.join(train_not_cat_path, os.path.basename(img_path)))
    for img_path in test_not_cats:
        shutil.copy(img_path, os.path.join(test_not_cat_path, os.path.basename(img_path)))

    print(f"Processed not cat class {len(train_not_cats)} training {len(test_not_cats)} testing images")
    print('\nDataset preparation complete')


if __name__ == '__main__':
    # --- IMPORTANT: CONFIGURE YOUR PATHS HERE ---
    SOURCE_ANIMAL_DATASET_DIR = 'others_animal' # The folder with 80+ animal subfolders
    DESTINATION_DATASET_DIR = 'cat_vs_not_cat_dataset'   # The new folder that will be created
    # -------------------------------------------
    
    if not os.path.isdir(SOURCE_ANIMAL_DATASET_DIR):
        print(f"Error: Source directory not found at '{SOURCE_ANIMAL_DATASET_DIR}'")
        print("Please update the SOURCE_ANIMAL_DATASET_DIR variable in the script.")
    else:
        prepare_binary_dataset(SOURCE_ANIMAL_DATASET_DIR, DESTINATION_DATASET_DIR)
