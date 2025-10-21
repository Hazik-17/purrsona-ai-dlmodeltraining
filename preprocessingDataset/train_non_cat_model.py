import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report

# === Configuration ===
DATA_DIR = 'cat_vs_not_cat_dataset'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-4

# === Data Generators ===
# Use moderate augmentation, as this is a simpler task
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.2,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary', # For two classes
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'test'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print("\nClass indices:", train_generator.class_indices) # e.g., {'cat': 0, 'not_cat': 1}

# === Model Setup ===
# Load MobileNetV3Small pre-trained on ImageNet
base_model = MobileNetV3Small(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights='imagenet'
)
# Freeze the base model layers
base_model.trainable = False

# Add our custom classifier head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
# The final layer for binary classification
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# === Compile the Model ===
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy', # Loss function for two classes
    metrics=['accuracy']
)

model.summary()

# === Callbacks ===
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5, min_lr=1e-7)

# === Train the Model ===
print("\n--- Starting non_cat Model Training ---")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    callbacks=[early_stop, reduce_lr]
)

# === Final Evaluation ===
print("\n--- Evaluating non_cat Model on Test Set ---")
loss, accuracy = model.evaluate(test_generator, verbose=1)
print(f"✅ Test Accuracy: {accuracy * 100:.2f}%")

# Detailed classification report
test_generator.reset()
y_pred_probs = model.predict(test_generator)
y_pred = np.round(y_pred_probs).astype(int).flatten()
y_true = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

# === Save the Final Model ===
model.save('cat_vs_not_cat.keras')
print("\n💾 non_cat model saved as 'cat_vs_not_cat.keras'")

# Also save the class indices for the test script
with open('non_cat_class_indices.json', 'w') as f:
    import json
    json.dump(train_generator.class_indices, f)
print("💾 Non_cat class indices saved as 'non_cat_class_indices.json'")
