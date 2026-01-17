import os
import glob
import json
import numpy as np
import tensorflow as tf

# Import EfficientNetV2B0 and the matching image preprocessor
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
# ---

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import sys

# Turn on mixed precision using mixed_float16
print("--- Enabling Mixed Precision mixed_float16 ---")
tf.keras.mixed_precision.set_global_policy('mixed_float16')
AUTOTUNE = tf.data.AUTOTUNE

# Check if a GPU is available and set memory growth
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

# Training settings and file locations
DATA_DIR = '../../output'
IMG_SIZE = (224, 224) # image size used for the model
# Batch size used for training
BATCH_SIZE = 32
EPOCHS_PHASE1 = 20
EPOCHS_PHASE2 = 30
LR_PHASE1 = 1e-3
LR_PHASE2 = 8e-6 # small rate for fine tuning
EARLYSTOP_PATIENCE = 10
LABEL_SMOOTHING = 0.1
DROPOUT_RATE = 0.5
DENSE_UNITS = 256
UNFREEZE_LAYERS = 50

# Set up the data pipeline for training validation and test data
print("\n--- Setting up tf.data.Dataset pipeline ---")

# This function gets an image and a label
# It makes random image changes and runs the model preprocessor
# Input  image tensor and label tensor
# Output image tensor and the same label
def augment_and_preprocess(image, label):
    # Flip the image left or right
    image = tf.image.random_flip_left_right(image)
    # Flip the image up or down
    image = tf.image.random_flip_up_down(image)

    image_shape = tf.shape(image)
    crop_size = tf.cast(tf.cast(image_shape[:2], tf.float32) * tf.random.uniform(shape=[], minval=0.6, maxval=1.0), tf.int32)
    image = tf.image.random_crop(image, size=[crop_size[0], crop_size[1], 3])
    image = tf.image.resize(image, [IMG_SIZE[0], IMG_SIZE[1]])
    image = tf.image.random_brightness(image, max_delta=0.4)

    # Run EfficientNetV2 style preprocessor on the image
    image = preprocess_input(image)
    return image, label

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

# Make fast versions of the training validation and test sets
print("Configuring training pipeline train_ds")
train_ds = train_ds_raw.map(augment_and_preprocess, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.shuffle(buffer_size=1024)
train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

print("Configuring validation pipeline val_ds")
# Apply the model preprocessor to validation images
val_ds = val_ds_raw.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

print("Configuring test pipeline test_ds")
# Apply the model preprocessor to test images
test_ds = test_ds_raw.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

# --- Get class names and save class_indices.json ---
class_names = train_ds_raw.class_names
num_classes = len(class_names)
class_indices = {name: i for i, name in enumerate(class_names)}
print(f"\nFound {num_classes} classes: {class_indices}")

print("\n💾 Saving class indices to breed_expert_class_indices_effnet.json")
with open('breed_expert_class_indices_effnet.json', 'w') as f:
    json.dump(class_indices, f)
# --- End of tf.data setup ---


# Build the model using EfficientNetV2B0 as the base
base_model = EfficientNetV2B0(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights='imagenet'
)
# ---
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

# Phase 1 train only the top new layers
print("\n🔁 Starting Phase 1 train top layers only EfficientNetV2B0")
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
plt.savefig('phase_1_training_plot_effnet.png'); print("\nSaved Phase 1 plot as 'phase_1_training_plot_effnet.png'")

# Phase 2 fine tune the last layers of the base model
print(f"\n🔁 Starting Phase 2 fine tuning last {UNFREEZE_LAYERS} layers EfficientNetV2B0")
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
plt.savefig('phase_2_training_plot_effnet.png'); print("\nSaved Phase 2 plot as 'phase_2_training_plot_effnet.png'")

# Run final test set evaluation and show accuracy
print("\n📊 Final evaluation on test set EfficientNetV2B0")
loss, accuracy = model.evaluate(test_ds, verbose=2, steps=test_steps)
print(f"✅ Test accuracy: {accuracy * 100:.2f}%")

# Make a detailed classification report for the test set
print("\nGenerating Classification Report")
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

# Check training accuracy to see if model learned well on train set
print("\nEvaluating training accuracy")
train_loss, train_accuracy = model.evaluate(train_ds, verbose=2, steps=steps_per_epoch)
print(f"Training accuracy: {train_accuracy * 100:.2f}%")

# Save the final trained model to disk
model.save('breed_expert_efficientnetv2b0_v2.keras'); print("\n💾 Model saved as 'breed_expert_efficientnetv2b0_v2.keras'")

print("✅ Training complete and all files saved.")