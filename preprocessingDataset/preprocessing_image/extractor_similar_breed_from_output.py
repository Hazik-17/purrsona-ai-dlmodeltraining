import os
import shutil

# This script makes a small dataset with only a few similar breeds
# Input a full dataset folder with train and test subfolders
# Output a new folder with the same train and test layout for chosen breeds
SOURCE_DIR = 'output'
DEST_DIR = 'similar_breed_dataset'
SIMILAR_BREEDS = [
    'Birman',
    'British',
    'Maine',
    'Persian',
    'Ragdoll'
]


# This function copies only the chosen breed folders into a new dataset
# Input source folder and list of breeds to copy
# Output new dataset folder ready for training
def create_specialized_dataset():
    print('Creating specialized dataset for similar breeds')
    print(f'Source Directory: {SOURCE_DIR}')
    print(f'Destination Directory: {DEST_DIR}')

    if not os.path.isdir(SOURCE_DIR):
        print('Error source directory not found')
        return

    os.makedirs(DEST_DIR, exist_ok=True)

    for subset in ['train', 'test']:
        print('\nProcessing subset', subset)
        source_subset_path = os.path.join(SOURCE_DIR, subset)
        dest_subset_path = os.path.join(DEST_DIR, subset)
        os.makedirs(dest_subset_path, exist_ok=True)

        if not os.path.isdir(source_subset_path):
            print('Warning source subset folder not found', source_subset_path)
            continue

        for breed in SIMILAR_BREEDS:
            source_breed_path = os.path.join(source_subset_path, breed)
            dest_breed_path = os.path.join(dest_subset_path, breed)

            if os.path.isdir(source_breed_path):
                print('Copying breed', breed)
                if os.path.exists(dest_breed_path):
                    shutil.rmtree(dest_breed_path)
                shutil.copytree(source_breed_path, dest_breed_path)
            else:
                print('Warning breed folder not found', source_breed_path)

    print('\nSpecialized dataset created successfully')
    print('New dataset at', DEST_DIR)


if __name__ == '__main__':
    create_specialized_dataset()

