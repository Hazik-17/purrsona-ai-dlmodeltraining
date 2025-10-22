import os
import glob
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report

# === Configuration ===
DATA_DIR = 'cat_vs_not_cat_dataset'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_PHASE1 = 15
EPOCHS_PHASE2 = 25
LR_PHASE1 = 1e-4
LR_PHASE2 = 1e-6
UNFREEZE_LAYERS = 30

# === Data Generators ===
print("--- Setting up Data Generators for Gatekeeper Model ---")
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.2,
    fill_mode='nearest',
    validation_split=0.2  # Use part of the training data for validation
)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

valid_generator = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'test'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print("\nClass indices:", train_generator.class_indices)

# === Model Setup ===
print("\n--- Building the Gatekeeper Model ---")
base_model = MobileNetV3Small(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x) # Added Batch Normalization for stability
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# === Callbacks ===
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=1e-8)

# === Phase 1: Train the Head ===
print("\n--- Starting Phase 1: Training Top Layer ---")
model.compile(
    optimizer=Adam(learning_rate=LR_PHASE1),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
history1 = model.fit(
    train_generator,
    epochs=EPOCHS_PHASE1,
    validation_data=valid_generator,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# === Phase 2: Fine-Tune ===
print(f"\n--- Starting Phase 2: Fine-Tuning last {UNFREEZE_LAYERS} layers ---")
for layer in base_model.layers[-UNFREEZE_LAYERS:]:
    if not isinstance(layer, BatchNormalization):
        layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=LR_PHASE2),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
history2 = model.fit(
    train_generator,
    epochs=EPOCHS_PHASE2,
    validation_data=valid_generator,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# === Final Evaluation ===
print("\n--- Evaluating Non-cat Model on Test Set ---")
loss, accuracy = model.evaluate(test_generator, verbose=1)
print(f"✅ Test Accuracy: {accuracy * 100:.2f}%")

# Detailed classification report
y_pred_probs = model.predict(test_generator)
y_pred = (y_pred_probs > 0.5).astype("int32").flatten()
y_true = test_generator.classes
class_labels = list(train_generator.class_indices.keys())

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

# === Save the Final Model and Labels ===
model.save('cat_vs_not_cat.keras')
print("\n💾 Non-cat model saved as 'cat_vs_not_cat.keras'")

with open('non_cat_class_indices.json', 'w') as f:
    json.dump(train_generator.class_indices, f)
print("💾 Non-cat class indices saved as 'non_cat_class_indices.json'")
