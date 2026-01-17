import os
import glob
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# === Configuration (Using "Run 4.0A" settings on the new dataset) ===
DATA_DIR = 'similar_breed_dataset'
# ---

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS_PHASE1 = 20
EPOCHS_PHASE2 = 30
LR_PHASE1 = 1e-3
LR_PHASE2 = 8e-6
EARLYSTOP_PATIENCE = 10
LABEL_SMOOTHING = 0.1
DROPOUT_RATE = 0.5
DENSE_UNITS = 256
UNFREEZE_LAYERS = 50

# === Data Generators (using the same aggressive augmentation) ===
print("--- Setting up Data Generators for Similar Breeds ---")
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    zoom_range=0.4,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.6, 1.4],
    horizontal_flip=True,
    vertical_flip=True,
    shear_range=0.3,
    fill_mode='nearest',
    validation_split=0.2
)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

valid_generator = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'test'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# This will now automatically detect 5 classes instead of 12
print("\nDetected Class Indices:", train_generator.class_indices)
num_classes = len(train_generator.class_indices)

# === Model Setup (Will now have a 5-class output layer) ===
print("\n--- Building the Similar Breed Expert Model ---")
base_model = MobileNetV3Small(
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
    Dense(num_classes, activation='softmax') # Automatically creates a Dense(5,...) layer
])

model.summary()

# === Callbacks ===
early_stop = EarlyStopping(monitor='val_loss', patience=EARLYSTOP_PATIENCE, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr=1e-8, verbose=1)

# === Phase 1: Train Classifier Head ===
print("\n--- Starting Phase 1: Training Expert Classifier Head ---")
model.compile(
    optimizer=Adam(learning_rate=LR_PHASE1),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
    metrics=['accuracy']
)
history1 = model.fit(
    train_generator,
    epochs=EPOCHS_PHASE1,
    validation_data=valid_generator,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# === Phase 2: Fine-Tune Top Layers ===
print(f"\n--- Starting Phase 2: Fine-Tuning Top {UNFREEZE_LAYERS} Layers ---")
for layer in base_model.layers[-UNFREEZE_LAYERS:]:
    if not isinstance(layer, BatchNormalization):
        layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=LR_PHASE2, clipnorm=1.0),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
    metrics=['accuracy']
)
history2 = model.fit(
    train_generator,
    epochs=EPOCHS_PHASE2,
    validation_data=valid_generator,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# === Final Evaluation on Test Set ===
print("\n--- Final Evaluation of Expert Model on Test Set ---")
loss, accuracy = model.evaluate(test_generator, verbose=1)
print(f"✅ Expert Model Test Accuracy: {accuracy * 100:.2f}%")

# Detailed Classification Report
test_generator.reset()
y_pred_probs = model.predict(test_generator, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

print("\nExpert Model Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels, digits=4))

# === Save the Final Expert Model and its Labels ===
FINAL_MODEL_NAME = 'similar_breed_expert.keras'
FINAL_LABELS_NAME = 'similar_breed_class_indices.json'

model.save(FINAL_MODEL_NAME)
print(f"\n💾 Expert model saved as '{FINAL_MODEL_NAME}'")

with open(FINAL_LABELS_NAME, 'w') as f:
    json.dump(train_generator.class_indices, f)
print(f"💾 Expert class indices saved as '{FINAL_LABELS_NAME}'")
