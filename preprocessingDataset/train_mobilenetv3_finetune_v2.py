import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd

# ===== Config =====
TRAIN_DIR = 'output/train'
TEST_DIR = 'output/test'
LABELS_CSV = 'output/labels.csv'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_PHASE1 = 5
EPOCHS_PHASE2 = 15
LEARNING_RATE_PHASE1 = 0.001
LEARNING_RATE_PHASE2 = 0.0001

# ===== Load labels =====
df = pd.read_csv(LABELS_CSV)
breeds = sorted(df['breed'].unique())
num_classes = len(breeds)

# ===== Image Augmentation =====
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.05,
    height_shift_range=0.05,
    brightness_range=[0.9, 1.1],
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# ===== After train_generator is created =====
print("🔍 Keras class indices (from directory structure):")
for breed, idx in train_generator.class_indices.items():
    print(f"{breed}: {idx}")

# ===== Build Model =====
base_model = MobileNetV3Small(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze first

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=LEARNING_RATE_PHASE1),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# ===== Phase 1: Train Top Layers =====
print("🔁 Phase 1: Training top layers only...")
model.fit(
    train_generator,
    epochs=EPOCHS_PHASE1,
    validation_data=test_generator,
    callbacks=[early_stop]
)

# ===== Phase 2: Fine-tune full model =====
print("🔁 Phase 2: Unfreezing base model and fine-tuning...")
base_model.trainable = True
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE_PHASE2),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(
    train_generator,
    epochs=EPOCHS_PHASE2,
    validation_data=test_generator,
    callbacks=[early_stop]
)

# ===== Final Evaluation =====
print("\n📊 Final Evaluation:")
loss, acc = model.evaluate(test_generator)
print(f"✅ Test Accuracy: {acc*100:.2f}%")

# ===== Save Model =====
model.save('cat_breed_model_finetuned_v2.h5')
print("💾 Saved as 'cat_breed_model_finetuned_v2.h5'")
