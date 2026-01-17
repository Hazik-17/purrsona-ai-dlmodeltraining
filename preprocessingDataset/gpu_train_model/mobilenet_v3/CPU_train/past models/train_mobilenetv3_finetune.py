import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


# === Configuration ===
DATA_DIR = 'output_3breeds'   # Update as needed
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS_PHASE1 = 20
EPOCHS_PHASE2 = 30
LR_PHASE1 = 1e-3
LR_PHASE2 = 1e-6
EARLYSTOP_PATIENCE = 10

# === Check class distributions and consistency ===
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
assert train_classes == test_classes, "Mismatch in train/test classes!"

# === Data Generators ===
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.3,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    vertical_flip=False,
    shear_range=0.2,
    fill_mode='nearest',
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

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
    shuffle=False
)

print("\nClass indices:", train_generator.class_indices)

num_classes = len(train_generator.class_indices)

# === Model setup ===
base_model = MobileNetV3Small(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=LR_PHASE1),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.0),  # No label smoothing initially
    metrics=['accuracy']
)

early_stop = EarlyStopping(monitor='val_loss', patience=EARLYSTOP_PATIENCE, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr=1e-8, verbose=1)

# === Phase 1: Train top layers ===
print("\n🔁 Starting Phase 1: Train top layers")
history1 = model.fit(
    train_generator,
    epochs=EPOCHS_PHASE1,
    validation_data=valid_generator,
    callbacks=[early_stop, reduce_lr],
    verbose=2
)

# Plot Phase 1 accuracy and loss
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

# === Phase 2: Fine-tune last 30 layers of base_model ===
print("\n🔁 Starting Phase 2: Fine-tuning last 30 layers")
for layer in base_model.layers[:-30]:
    layer.trainable = False
for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=LR_PHASE2, clipnorm=1.0),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_generator,
    epochs=EPOCHS_PHASE2,
    validation_data=valid_generator,
    callbacks=[early_stop, reduce_lr],
    verbose=2
)

# Plot Phase 2 accuracy and loss
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

# === Evaluate on Test Set ===
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

# === Optional: Evaluate on training data for overfitting check ===
print("\nEvaluating training accuracy (optional check):")
train_loss, train_accuracy = model.evaluate(train_generator, verbose=2)
print(f"Training accuracy: {train_accuracy * 100:.2f}%")

# === Save the trained model ===
model.save('cat_breed_mobilenetv3_finetuned_corrected.h5')
print("\n💾 Model saved as 'cat_breed_mobilenetv3_finetuned_corrected.h5'")
