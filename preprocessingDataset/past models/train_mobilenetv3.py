import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
import pandas as pd

# ===== Paths =====
TRAIN_DIR = 'output/train'
TEST_DIR = 'output/test'
LABELS_CSV = 'output/labels.csv'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# ===== Load label mappings =====
df = pd.read_csv(LABELS_CSV)
breeds = sorted(df['breed'].unique())
num_classes = len(breeds)

# ===== Data Generators =====
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=15
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

# ===== Build Model =====
base_model = MobileNetV3Small(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze base

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.0005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ===== Train =====
model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator
)

# ===== Evaluate =====
print("\n🔍 Final Evaluation:")
loss, acc = model.evaluate(test_generator)
print(f"Test Accuracy: {acc*100:.2f}%")

# ===== Save Model =====
model.save('cat_breed_model.h5')
print("✅ Model saved as 'cat_breed_model.h5'")
