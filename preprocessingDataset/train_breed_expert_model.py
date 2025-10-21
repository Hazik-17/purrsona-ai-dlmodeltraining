import os
import glob
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

# === Configuration ===
DATA_DIR = 'output'
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

# === Check class distributions ===
def print_class_distribution(data_dir, subset):
    print(f"\nClass distribution in {subset}:")
    path = os.path.join(data_dir, subset)
    classes = sorted(os.listdir(path))
    for c in classes:
        count = len(glob.glob(os.path.join(path, c, '*.jpg'))) + len(glob.glob(os.path.join(path, c, '*.png')))
        print(f"  {c}: {count} images")
    return classes

train_classes = print_class_distribution(DATA_DIR, 'train')
test_classes = print_class_distribution(DATA_DIR, 'test')
assert train_classes == test_classes, "Mismatch in train/test class folders!"

# === Data Generators with MobileNetV3 preprocessing ===
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,         # <--- INCREASED from 25
    zoom_range=0.4,            # <--- INCREASED from 0.3
    width_shift_range=0.2,     # <--- INCREASED from 0.1
    height_shift_range=0.2,    # <--- INCREASED from 0.1
    brightness_range=[0.6, 1.4], # <--- INCREASED range
    horizontal_flip=True,
    vertical_flip=True,        
    shear_range=0.3,           # <--- INCREASED from 0.2
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
    shuffle=True,
    seed=42
)

valid_generator = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=True,
    seed=42
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'test'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False  # Important for label matching during evaluation
)

print("\nClass indices:", train_generator.class_indices)

# === Debug print: check a batch of images and labels ===
images, labels = next(train_generator)
print(f"Train batch images shape: {images.shape}")  # should be (batch_size, 224, 224, 3)
print(f"Train batch labels shape: {labels.shape}")  # should be (batch_size, num_classes)
print(f"First training label (one-hot): {labels[0]}")

# === Model Setup ===
num_classes = len(train_generator.class_indices)

base_model = MobileNetV3Small(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(DENSE_UNITS, activation='relu', kernel_regularizer=l2(0.001)), # <--- CHANGE
    BatchNormalization(),
    Dropout(DROPOUT_RATE),                                            
    Dense(num_classes, activation='softmax')
])

early_stop = EarlyStopping(monitor='val_loss', patience=EARLYSTOP_PATIENCE, restore_best_weights=True) 
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr=1e-8, verbose=1) # <--- CHANGE

# === Phase 1: Train top layers only ===
print("\n🔁 Starting Phase 1: Train top layers only")
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
    verbose=2
)

# Plot Phase 1 results
plt.figure(figsize=(12, 5))
plt.subplot(1,2,1)
plt.plot(history1.history['accuracy'], label='train_acc')
plt.plot(history1.history['val_accuracy'], label='val_acc')
plt.title('Phase 1 Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history1.history['loss'], label='train_loss')
plt.plot(history1.history['val_loss'], label='val_loss')
plt.title('Phase 1 Loss')
plt.legend()
plt.show()

# === Phase 2: Fine-tune last 50 layers ===
print(f"\n🔁 Starting Phase 2: Fine-tuning last {UNFREEZE_LAYERS} layers")
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
    verbose=2
)


# Plot Phase 2 results
plt.figure(figsize=(12, 5))
plt.subplot(1,2,1)
plt.plot(history2.history['accuracy'], label='train_acc')
plt.plot(history2.history['val_accuracy'], label='val_acc')
plt.title('Phase 2 Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history2.history['loss'], label='train_loss')
plt.plot(history2.history['val_loss'], label='val_loss')
plt.title('Phase 2 Loss')
plt.legend()
plt.show()

# === Final Evaluation on Test Set ===
print("\n📊 Final evaluation on test set:")
loss, accuracy = model.evaluate(test_generator, verbose=2)
print(f"✅ Test accuracy: {accuracy * 100:.2f}%")

# Detailed classification report
test_generator.reset()
y_pred_probs = model.predict(test_generator, verbose=2)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels, digits=4))

# === Optional: Check training accuracy to diagnose underfitting ===
print("\nEvaluating training accuracy:")
train_loss, train_accuracy = model.evaluate(train_generator, verbose=2)
print(f"Training accuracy: {train_accuracy * 100:.2f}%")

# === Save the trained model ===
model.save('breed_expert_mobilenetV3.keras')
print("\n💾 Model saved as 'breed_expert_mobilenetV3.keras'")

# === Add this to the end of your training script ===
import json

print("\n💾 Saving class indices to breed_expert_class_indices.json")
class_indices = train_generator.class_indices
with open('breed_expert_class_indices.json', 'w') as f:
    json.dump(class_indices, f)

print("✅ Training complete and all files saved.")