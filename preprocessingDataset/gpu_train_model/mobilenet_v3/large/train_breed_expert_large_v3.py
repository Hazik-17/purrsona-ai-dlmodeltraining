import os
import glob
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import sys

# === 1. GPU Configuration & Mixed Precision ===
print("--- Enabling Mixed Precision (mixed_float16) ---")
tf.keras.mixed_precision.set_global_policy('mixed_float16')
AUTOTUNE = tf.data.AUTOTUNE

print("--- Checking for GPU ---")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"✅ Found {len(gpus)} physical GPUs, {len(logical_gpus)} logical GPUs.")
        print("   GPU will be used for training.")
    except RuntimeError as e:
        print(f"⚠️ Could not configure GPU: {e}")
else:
    print("⚠️ WARNING: No GPU found by TensorFlow. Training will be slow.")
print("-------------------------")

# === 2. Configuration (Run 8.0 settings + BATCH_SIZE=32) ===
DATA_DIR = '../../output'
IMG_SIZE = (224, 224) 
# --- CHANGE: Using BATCH_SIZE as a regularizer ---
BATCH_SIZE = 32 # Was 64. This is our final test.
# ---
EPOCHS_PHASE1 = 20
EPOCHS_PHASE2 = 30
LR_PHASE1 = 1e-3
LR_PHASE2 = 8e-6     # Our champion fine-tuning LR
EARLYSTOP_PATIENCE = 10
LABEL_SMOOTHING = 0.1 
DROPOUT_RATE = 0.5
DENSE_UNITS = 256
UNFREEZE_LAYERS = 50

# === 3. tf.data.Dataset Pipeline ===
print("\n--- Setting up tf.data.Dataset pipeline ---")

# --- Helper functions ---
def augment_and_preprocess(image, label):
    """Applies data augmentation and MobileNetV3 preprocessing."""
    # Data Augmentation
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    
    image_shape = tf.shape(image)
    crop_size = tf.cast(tf.cast(image_shape[:2], tf.float32) * tf.random.uniform(shape=[], minval=0.6, maxval=1.0), tf.int32)
    image = tf.image.random_crop(image, size=[crop_size[0], crop_size[1], 3])
    image = tf.image.resize(image, [IMG_SIZE[0], IMG_SIZE[1]])
    image = tf.image.random_brightness(image, max_delta=0.4)
    
    image = preprocess_input(image)
    return image, label
# ---

# Create the initial training dataset from directories
train_ds_raw = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, 'train'),
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=None,
    label_mode='categorical'
)

# Create the validation dataset
val_ds_raw = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, 'train'),
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=None,
    label_mode='categorical'
)

# Create the test dataset
test_ds_raw = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, 'test'),
    image_size=IMG_SIZE,
    batch_size=None,
    label_mode='categorical',
    shuffle=False
)

# --- Manually calculate dataset sizes and steps ---
print("Calculating dataset sizes...")
train_size = train_ds_raw.cardinality().numpy()
val_size = val_ds_raw.cardinality().numpy()
test_size = test_ds_raw.cardinality().numpy()

steps_per_epoch = int(np.ceil(train_size / BATCH_SIZE))
validation_steps = int(np.ceil(val_size / BATCH_SIZE))
test_steps = int(np.ceil(test_size / BATCH_SIZE))

print(f"  - BATCH_SIZE: {BATCH_SIZE}")
print(f"  - Training images: {train_size}  -> Steps per epoch: {steps_per_epoch}")
print(f"  - Validation images: {val_size} -> Validation steps: {validation_steps}")
print(f"  - Test images: {test_size}     -> Test steps: {test_steps}")
# ---

# --- Create the high-performance data pipelines ---
print("Configuring training pipeline (train_ds)...")
train_ds = train_ds_raw.map(augment_and_preprocess, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.shuffle(buffer_size=1024)
train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

print("Configuring validation pipeline (val_ds)...")
val_ds = val_ds_raw.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

print("ConfigCuring test pipeline (test_ds)...")
test_ds = test_ds_raw.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

# --- Get class names and save class_indices.json ---
class_names = train_ds_raw.class_names
num_classes = len(class_names)
class_indices = {name: i for i, name in enumerate(class_names)}
print(f"\nFound {num_classes} classes: {class_indices}")

print("\n💾 Saving class indices to breed_expert_class_indices.json")
with open('breed_expert_class_indices.json', 'w') as f:
    json.dump(class_indices, f)
# --- End of tf.data setup ---


# === 4. Model Setup ===
base_model = MobileNetV3Large(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(DENSE_UNITS, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(DROPOUT_RATE),
    Dense(num_classes, activation='softmax', dtype='float32')
])

early_stop = EarlyStopping(monitor='val_loss', patience=EARLYSTOP_PATIENCE, restore_best_weights=True) 
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr=1e-8, verbose=1)

# === 5. Phase 1: Train top layers only ===
print("\n🔁 Starting Phase 1: Train top layers only (MobileNetV3Large)")
model.compile(
    optimizer=Adam(learning_rate=LR_PHASE1),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
    metrics=['accuracy']
)

history1 = model.fit(
    train_ds,
    epochs=EPOCHS_PHASE1,
    validation_data=val_ds,
    callbacks=[early_stop, reduce_lr],
    verbose=2,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps
)

# Plot Phase 1 results
plt.figure(figsize=(12, 5))
plt.subplot(1,2,1); plt.plot(history1.history['accuracy'], label='train_acc'); plt.plot(history1.history['val_accuracy'], label='val_acc'); plt.title('Phase 1 Accuracy'); plt.legend()
plt.subplot(1,2,2); plt.plot(history1.history['loss'], label='train_loss'); plt.plot(history1.history['val_loss'], label='val_loss'); plt.title('Phase 1 Loss'); plt.legend()
plt.savefig('phase_1_training_plot_large_v3.png'); print("\nSaved Phase 1 plot as 'phase_1_training_plot_large_v3.png'")

# === 6. Phase 2: Fine-tune last 50 layers ===
print(f"\n🔁 Starting Phase 2: Fine-tuning last {UNFREEZE_LAYERS} layers (MobileNetV3Large)")
for layer in base_model.layers[-UNFREEZE_LAYERS:]: 
    if not isinstance(layer, BatchNormalization):
        layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=LR_PHASE2, clipnorm=1.0), 
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
    metrics=['accuracy']
)

history2 = model.fit(
    train_ds,
    epochs=EPOCHS_PHASE2,
    validation_data=val_ds,
    callbacks=[early_stop, reduce_lr],
    verbose=2,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps
)

# Plot Phase 2 results
plt.figure(figsize=(12, 5));
plt.subplot(1,2,1); plt.plot(history2.history['accuracy'], label='train_acc'); plt.plot(history2.history['val_accuracy'], label='val_acc'); plt.title('Phase 2 Accuracy'); plt.legend()
plt.subplot(1,2,2); plt.plot(history2.history['loss'], label='train_loss'); plt.plot(history2.history['val_loss'], label='val_loss'); plt.title('Phase 2 Loss'); plt.legend()
plt.savefig('phase_2_training_plot_large_v3.png'); print("\nSaved Phase 2 plot as 'phase_2_training_plot_large_v3.png'")

# === 7. Final Evaluation on Test Set ===
print("\n📊 Final evaluation on test set (MobileNetV3Large):")
loss, accuracy = model.evaluate(test_ds, verbose=2, steps=test_steps)
print(f"✅ Test accuracy: {accuracy * 100:.2f}%")

# Detailed classification report
print("\nGenerating Classification Report...")
y_pred_probs = model.predict(test_ds, verbose=2, steps=test_steps)
y_pred = np.argmax(y_pred_probs, axis=1)

y_true = []
for images, labels in test_ds.take(test_steps):
    y_true.append(np.argmax(labels.numpy(), axis=1))
y_true = np.concatenate(y_true)

if len(y_pred) > len(y_true):
    y_pred = y_pred[:len(y_true)]

class_labels = list(class_indices.keys())
print("\nClassification Report:"); print(classification_report(y_true, y_pred, target_names=class_labels, digits=4))

# === Optional: Check training accuracy to diagnose underfitting ===
print("\nEvaluating training accuracy:")
train_loss, train_accuracy = model.evaluate(train_ds, verbose=2, steps=steps_per_epoch)
print(f"Training accuracy: {train_accuracy * 100:.2f}%")

# === 8. Save the trained model ===
model.save('breed_expert_mobilenetv3_large_v3.keras'); print("\n💾 Model saved as 'breed_expert_mobilenetv3_large_v3.keras'")

print("✅ Training complete and all files saved.")