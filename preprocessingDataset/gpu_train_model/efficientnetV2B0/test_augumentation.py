import tensorflow as tf
import matplotlib.pyplot as plt
import os

# --- 1. CONFIGURATION ---
IMG_SIZE = (224, 224)
DATA_DIR = '../../output/train'  # Ensure this path is correct
CLASS_NAME = 'Persian'           # Change breed if needed
SAMPLE_INDEX = 0                 # Change to test different images

# --- 2. AUGMENTATION FUNCTIONS (ISOLATED) ---

def apply_flip(image):
    # Technique A: Random Spatial Flips
    # We force a flip here for visualization purposes
    image = tf.image.flip_left_right(image)
    image = tf.image.flip_up_down(image)
    return image

def apply_crop_zoom(image):
    # Technique C: Random Crop and Zoom
    image_shape = tf.shape(image)
    # Force a visible crop (70%)
    crop_size = tf.cast(
        tf.cast(image_shape[:2], tf.float32) * 0.7, 
        tf.int32
    )
    image = tf.image.random_crop(image, size=[crop_size[0], crop_size[1], 3])
    # Resize back (Zoom effect)
    image = tf.image.resize(image, [IMG_SIZE[0], IMG_SIZE[1]])
    return image

def apply_brightness(image):
    # Technique D: Random Brightness
    # Force a visible brightness change
    image = tf.image.adjust_brightness(image, delta=0.3)
    return image

def apply_all_combined(image):
    # Apply all strategies together (This is what the model actually sees)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    
    image_shape = tf.shape(image)
    crop_size = tf.cast(
        tf.cast(image_shape[:2], tf.float32) * tf.random.uniform(shape=[], minval=0.6, maxval=1.0), 
        tf.int32
    )
    image = tf.image.random_crop(image, size=[crop_size[0], crop_size[1], 3])
    image = tf.image.resize(image, [IMG_SIZE[0], IMG_SIZE[1]])
    image = tf.image.random_brightness(image, max_delta=0.4)
    return image

# --- 3. HELPER: LOAD IMAGE ---
def load_sample_image(data_dir, class_name, index):
    class_path = os.path.join(data_dir, class_name)
    if not os.path.exists(class_path):
        raise ValueError(f"Directory not found: {class_path}")
        
    image_files = os.listdir(class_path)
    if not image_files:
        raise ValueError("No images found in directory.")
        
    img_path = os.path.join(class_path, image_files[index])
    print(f"Loading: {img_path}")
    
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    # Initial resize to ensure consistent starting point
    img = tf.image.resize(img, [IMG_SIZE[0], IMG_SIZE[1]])
    return img

# --- 4. EXECUTE VISUALIZATION ---
try:
    original_img = load_sample_image(DATA_DIR, CLASS_NAME, SAMPLE_INDEX)
    
    # Define the list of visualizations to generate
    # Format: (Title, Function_to_Run)
    augmentations = [
        ("Original\n(Pixel Rescaling)", lambda x: x), # No change, just resized
        ("Technique A:\nSpatial Flips", apply_flip),
        ("Technique C:\nCrop & Zoom", apply_crop_zoom),
        ("Technique D:\nBrightness", apply_brightness),
        ("Combined Strategy\n(Training Input)", apply_all_combined)
    ]

    plt.figure(figsize=(16, 5))
    plt.suptitle(f"Data Augmentation Techniques Analysis: {CLASS_NAME}", fontsize=16, weight='bold')

    for i, (title, func) in enumerate(augmentations):
        # 1. Apply the specific function
        aug_img = func(original_img)
        
        # 2. Clip values to 0-255 for valid display (matplotlib doesn't like unbounded floats)
        viz_img = tf.clip_by_value(aug_img, 0, 255)
        
        # 3. Plot
        plt.subplot(1, 5, i + 1)
        plt.imshow(viz_img.numpy().astype("uint8"))
        plt.title(title, fontsize=11)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig('augmentation_analysis.png')
    print("------------------------------------------------------")
    print("SUCCESS: Image saved as 'augmentation_analysis.png'")
    print("------------------------------------------------------")

except Exception as e:
    print(f"Error: {e}")