import os
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
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# -------------------------
# 1) Setup & config
# -------------------------
print("Using mixed_float16 policy")
tf.keras.mixed_precision.set_global_policy('mixed_float16')
AUTOTUNE = tf.data.AUTOTUNE

print("Checking for GPU...")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"Found {len(gpus)} physical GPUs, {len(logical_gpus)} logical GPUs.")
    except RuntimeError as e:
        print(f"Could not configure GPU: {e}")
else:
    print("No GPU found. Training may be slow.")

# === Hyperparameters (my original) ===
DATA_DIR = '../../../output'
IMG_SIZE = (224, 224)
BATCH_SIZE = 64
EPOCHS_PHASE1 = 20
EPOCHS_PHASE2 = 30
LR_PHASE1 = 1e-3
LR_PHASE2 = 8e-6
EARLYSTOP_PATIENCE = 10
LABEL_SMOOTHING = 0.1
DROPOUT_RATE = 0.5
DENSE_UNITS = 256
UNFREEZE_LAYERS = 50

# -------------------------
# 2) Data pipeline
# -------------------------
print("\n--- Setting up tf.data.Dataset pipeline ---")

def augment_and_preprocess(image, label):
    # same augmentation + preprocess_input
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image_shape = tf.shape(image)
    crop_size = tf.cast(tf.cast(image_shape[:2], tf.float32) * tf.random.uniform(shape=[], minval=0.6, maxval=1.0), tf.int32)
    image = tf.image.random_crop(image, size=[crop_size[0], crop_size[1], 3])
    image = tf.image.resize(image, [IMG_SIZE[0], IMG_SIZE[1]])
    image = tf.image.random_brightness(image, max_delta=0.4)
    image = preprocess_input(image)
    return image, label

# Load datasets (batch_size=None so we can create pipeline consistently)
train_ds_raw = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, 'train'),
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

# dataset sizes
train_size = train_ds_raw.cardinality().numpy()
val_size = val_ds_raw.cardinality().numpy()
test_size = test_ds_raw.cardinality().numpy()
steps_per_epoch = int(np.ceil(train_size / BATCH_SIZE))
validation_steps = int(np.ceil(val_size / BATCH_SIZE))

print(f"Train images: {train_size}, Val images: {val_size}, Test images: {test_size}")
print(f"Steps per epoch: {steps_per_epoch}, Validation steps: {validation_steps}")

# Build pipelines (with consistent preprocess_input)
train_ds = train_ds_raw.map(augment_and_preprocess, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.shuffle(buffer_size=1024).batch(BATCH_SIZE).prefetch(AUTOTUNE)

val_ds = val_ds_raw.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

test_ds = test_ds_raw.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

# Save class indices reliably (these are in sorted order as returned by image_dataset_from_directory)
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
print("Phase 1: training top layers")
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

plt.figure(figsize=(12,5))
plt.subplot(1,2,1); plt.plot(history1.history.get('accuracy',[]), label='train_acc'); plt.plot(history1.history.get('val_accuracy',[]), label='val_acc'); plt.legend(); plt.title('Phase1 Acc')
plt.subplot(1,2,2); plt.plot(history1.history.get('loss',[]), label='train_loss'); plt.plot(history1.history.get('val_loss',[]), label='val_loss'); plt.legend(); plt.title('Phase1 Loss')
plt.savefig('phase_1_training_plot_large.png')
print("Saved Phase 1 plot.")

# -------------------------
# 5) Phase 2 fine-tune
# -------------------------
print(f"Phase 2: fine-tuning last {UNFREEZE_LAYERS} layers")
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

plt.figure(figsize=(12,5))
plt.subplot(1,2,1); plt.plot(history2.history.get('accuracy',[]), label='train_acc'); plt.plot(history2.history.get('val_accuracy',[]), label='val_acc'); plt.legend(); plt.title('Phase2 Acc')
plt.subplot(1,2,2); plt.plot(history2.history.get('loss',[]), label='train_loss'); plt.plot(history2.history.get('val_loss',[]), label='val_loss'); plt.legend(); plt.title('Phase2 Loss')
plt.savefig('phase_2_training_plot_large.png')
print("Saved Phase 2 plot.")

# -------------------------
# 6) Final evaluation (robust)
# -------------------------
print("Final evaluation on test set")

# Evaluate using model.evaluate (uses whole dataset)
loss, accuracy = model.evaluate(test_ds, verbose=2)
print(f"Test accuracy (model.evaluate): {accuracy * 100:.2f}%  — loss: {loss:.4f}")

# Gather y_true and y_pred reliably (iterate until exhaustion)
y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(np.argmax(labels.numpy(), axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Basic checks
print("Test samples (y_true):", len(y_true))
print("Test predictions (y_pred):", len(y_pred))
unique_preds, counts = np.unique(y_pred, return_counts=True)
print("Prediction distribution (class_index:count):", dict(zip(unique_preds.tolist(), counts.tolist())))

# If model predicts only one class, this will show here and we can debug preprocessing/labels
if len(unique_preds) == 1:
    print("Warning: model predicted a single class for all test samples. This often means a preprocessing or label mismatch.")

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

# Confusion matrix plot (optional, saves to file)
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.xlabel('Predicted'); plt.ylabel('True')
plt.title('Confusion Matrix (Test)')
plt.tight_layout()
plt.savefig('confusion_matrix_test.png')
print("Saved confusion_matrix_test.png")

# -------------------------
# 7) Sanity: evaluate training set performance to detect under/overfit
# -------------------------
print("Evaluating training accuracy for sanity check (may take time)")
train_loss, train_accuracy = model.evaluate(train_ds, verbose=2)
print(f"Training accuracy: {train_accuracy * 100:.2f}%  — loss: {train_loss:.4f}")

# -------------------------
# 8) Save and reload to verify persistence
# -------------------------
model_path = 'breed_expert_mobilenetv3_large.keras'
model.save(model_path)
print(f"Model saved as '{model_path}'")

# reload and evaluate quickly to ensure saved weights are correct
reloaded = tf.keras.models.load_model(model_path)
loss_r, acc_r = reloaded.evaluate(test_ds, verbose=2)
print(f"Reloaded model test accuracy: {acc_r * 100:.2f}%  — loss: {loss_r:.4f}")

print("Training, evaluation, and saving complete.")
