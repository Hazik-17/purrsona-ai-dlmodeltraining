import os
import json
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# -------------------------
# 1) Setup & config
# -------------------------
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
    except RuntimeError as e:
        print(f"⚠️ Could not configure GPU: {e}")
else:
    print("⚠️ WARNING: No GPU found by TensorFlow. Training will be slow.")
print("-------------------------")

# === 🔧 Updated Hyperparameters ===
DATA_DIR = '../../../output'
IMG_SIZE = (224, 224)
BATCH_SIZE = 64
EPOCHS_PHASE1 = 20
EPOCHS_PHASE2 = 60  # 🔧 Increased from 40 → 60
LR_PHASE1 = 1e-3
LR_PHASE2 = 8e-6
EARLYSTOP_PATIENCE = 10
LABEL_SMOOTHING = 0.1  # 🔧 Increased for better generalization
DROPOUT_RATE = 0.5
DENSE_UNITS = 256
UNFREEZE_LAYERS = 125  # 🔧 Fine-tune deeper layers (was 75)

print("Using mixed_float16 policy")
# 2) Data pipeline
# -------------------------
print("\n--- Setting up tf.data.Dataset pipeline ---")
print("Checking for GPU...")
def augment_and_preprocess(image, label):
    # 🔧 Enhanced augmentations for variety
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    # Rotation and random zoom
          print(f"Found {len(gpus)} physical GPUs, {len(logical_gpus)} logical GPUs.")
    image = tfa.image.rotate(image, angle)
          print(f"Could not configure GPU: {e}")
    new_size = tf.cast(tf.cast(tf.shape(image)[:2], tf.float32) * scales, tf.int32)
     print("No GPU found. Training may be slow.")
    image = tf.image.resize_with_crop_or_pad(image, 224, 224)

    # 🔧 Stronger color augmentations
    image = tf.image.random_brightness(image, 0.5)
    image = tf.image.random_contrast(image, 0.5, 1.5)
    image = tf.image.random_saturation(image, 0.6, 1.4)
    image = tf.image.random_hue(image, 0.08)
    image = tf.image.random_jpeg_quality(image, 70, 100)

    # 🔧 Random crop for scale variation
    image = tf.image.random_crop(image, size=[200, 200, 3])
    image = tf.image.resize(image, [224, 224])

    # Final preprocess
    image = preprocess_input(image)
    return image, label

# Dataset loading
train_ds_raw = tf.keras.utils.image_dataset_from_directory(
     """Apply random augmentations and preprocess for MobileNetV3."""
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=None,
    label_mode='categorical'
)

val_ds_raw = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, 'train'),
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=None,
    label_mode='categorical'
)

test_ds_raw = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, 'test'),
    image_size=IMG_SIZE,
    batch_size=None,
    label_mode='categorical',
    shuffle=False
)

# Dataset sizes
train_size = train_ds_raw.cardinality().numpy()
val_size = val_ds_raw.cardinality().numpy()
test_size = test_ds_raw.cardinality().numpy()
steps_per_epoch = int(np.ceil(train_size / BATCH_SIZE))
validation_steps = int(np.ceil(val_size / BATCH_SIZE))

print(f"Train images: {train_size}, Val images: {val_size}, Test images: {test_size}")
print(f"Steps per epoch: {steps_per_epoch}, Validation steps: {validation_steps}")

# Pipelines
train_ds = train_ds_raw.map(augment_and_preprocess, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.shuffle(buffer_size=2048).batch(BATCH_SIZE).prefetch(AUTOTUNE)  # 🔧 Larger shuffle buffer

val_ds = val_ds_raw.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

test_ds = test_ds_raw.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

# Save class indices
class_names = train_ds_raw.class_names
num_classes = len(class_names)
class_indices = {name: i for i, name in enumerate(class_names)}
with open('breed_expert_class_indices.json', 'w') as f:
    json.dump(class_indices, f)
print("Saved class indices:", class_indices)

# -------------------------
# 3) Model setup
# -------------------------
base_model = MobileNetV3Large(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), include_top=False, weights='imagenet')
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

# -------------------------
# 4) Phase 1 training
# -------------------------
print("\n🔁 Phase 1: Training top layers")
model.compile(optimizer=Adam(learning_rate=LR_PHASE1),
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
              metrics=['accuracy'])

history1 = model.fit(
    train_ds,
    epochs=EPOCHS_PHASE1,
    validation_data=val_ds,
    callbacks=[early_stop, reduce_lr],
    verbose=2,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps
)

# Plot Phase 1
plt.figure(figsize=(12,5))
plt.subplot(1,2,1); plt.plot(history1.history['accuracy']); plt.plot(history1.history['val_accuracy']); plt.legend(['Train','Val']); plt.title('Phase 1 Accuracy')
plt.subplot(1,2,2); plt.plot(history1.history['loss']); plt.plot(history1.history['val_loss']); plt.legend(['Train','Val']); plt.title('Phase 1 Loss')
plt.savefig('phase_1_training_plot_large_v2.png')

# -------------------------
# 5) Phase 2 fine-tune
# -------------------------
print(f"\n🔁 Phase 2: Fine-tuning last {UNFREEZE_LAYERS} layers")
for layer in base_model.layers[-UNFREEZE_LAYERS:]:
    if not isinstance(layer, BatchNormalization):
        layer.trainable = True

model.compile(optimizer=Adam(learning_rate=LR_PHASE2, clipnorm=1.0),
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
              metrics=['accuracy'])

history2 = model.fit(
    train_ds,
    epochs=EPOCHS_PHASE2,
    validation_data=val_ds,
    callbacks=[early_stop, reduce_lr],
    verbose=2,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps
)

# Plot Phase 2
plt.figure(figsize=(12,5))
plt.subplot(1,2,1); plt.plot(history2.history['accuracy']); plt.plot(history2.history['val_accuracy']); plt.legend(['Train','Val']); plt.title('Phase 2 Accuracy')
plt.subplot(1,2,2); plt.plot(history2.history['loss']); plt.plot(history2.history['val_loss']); plt.legend(['Train','Val']); plt.title('Phase 2 Loss')
plt.savefig('phase_2_training_plot_large_v2.png')

# -------------------------
# 6) Final evaluation
# -------------------------
print("\n📊 Final evaluation on test set")

loss, accuracy = model.evaluate(test_ds, verbose=2)
print(f"✅ Test accuracy: {accuracy * 100:.2f}%  — loss: {loss:.4f}")

y_true, y_pred = [], []
for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(np.argmax(labels.numpy(), axis=1))

y_true, y_pred = np.array(y_true), np.array(y_pred)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.xlabel('Predicted'); plt.ylabel('True')
plt.title('Confusion Matrix (Test)')
plt.tight_layout()
plt.savefig('confusion_matrix_test_v2.png')

# -------------------------
# 7) Save model
# -------------------------
model.save('breed_expert_mobilenetv3_large_v2.keras')
print("💾 Saved model: breed_expert_mobilenetv3_large_v2.keras")

reloaded = tf.keras.models.load_model('breed_expert_mobilenetv3_large_v2.keras')
loss_r, acc_r = reloaded.evaluate(test_ds, verbose=2)
print(f"Reloaded model test accuracy: {acc_r * 100:.2f}%  — loss: {loss_r:.4f}")

print("\n✅ Training v2 complete with enhanced augmentations and fine-tuning.")
