import os
import shutil

# --- Configuration ---

# 1. The source directory containing your full 12-breed dataset.
#    Based on your previous scripts, this should be correct.
SOURCE_DIR = 'output'

# 2. The destination directory for our new, specialized dataset.
DEST_DIR = 'similar_breed_dataset'

# 3. The list of breeds we've identified as "similar" and want to extract.
#    These folder names MUST EXACTLY match the folder names in your SOURCE_DIR.
SIMILAR_BREEDS = [
    'Birman',
    'British',
    'Maine',
    'Persian',
    'Ragdoll'
]

def create_specialized_dataset():
    """
    Extracts specific breed folders from a full dataset to create a new,
    specialized dataset for training an expert model.
    """
    print(f"--- Creating Specialized Dataset for Similar Breeds ---")
    print(f"Source Directory: '{SOURCE_DIR}'")
    print(f"Destination Directory: '{DEST_DIR}'")

    if not os.path.isdir(SOURCE_DIR):
        print(f"❌ ERROR: Source directory '{SOURCE_DIR}' not found. Please check the path.")
        return

    # Create the base destination directory if it doesn't exist
    os.makedirs(DEST_DIR, exist_ok=True)

    # Loop through both 'train' and 'test' subsets
    for subset in ['train', 'test']:
        print(f"\nProcessing '{subset}' subset...")
        
        source_subset_path = os.path.join(SOURCE_DIR, subset)
        dest_subset_path = os.path.join(DEST_DIR, subset)

        # Create the train/test subdirectories in the destination
        os.makedirs(dest_subset_path, exist_ok=True)

        if not os.path.isdir(source_subset_path):
            print(f"  - ⚠️  Warning: Source subset folder '{source_subset_path}' not found. Skipping.")
            continue

        # Loop through our list of similar breeds
        for breed in SIMILAR_BREEDS:
            source_breed_path = os.path.join(source_subset_path, breed)
            dest_breed_path = os.path.join(dest_subset_path, breed)

            if os.path.isdir(source_breed_path):
                print(f"  - Copying '{breed}' images...")
                
                # If the destination folder already exists, remove it to ensure a clean copy
                if os.path.exists(dest_breed_path):
                    shutil.rmtree(dest_breed_path)
                
                # Copy the entire breed folder (including all images)
                shutil.copytree(source_breed_path, dest_breed_path)
            else:
                print(f"  - ⚠️  Warning: Breed folder '{source_breed_path}' not found. Skipping.")

    print("\n✅ Specialized dataset created successfully!")
    print(f"Your new dataset is ready at '{DEST_DIR}'. You can now use it to train the similar breed expert model.")

if __name__ == '__main__':
    create_specialized_dataset()

