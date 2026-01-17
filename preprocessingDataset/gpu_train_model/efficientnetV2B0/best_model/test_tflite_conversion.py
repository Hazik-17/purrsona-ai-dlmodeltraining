import tensorflow as tf

# Small helper script to test Keras -> TFLite conversion.
# It builds a tiny model, saves it, converts it to .tflite, and writes the file.

# Build a very small model for testing conversion only
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    tf.keras.layers.Conv2D(16, 3, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(2, activation='softmax')
])

# Save the Keras model file
model.save("dummy_model.keras")
print("Saved dummy_model.keras")

# Convert to TFLite format and save
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("dummy_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Saved dummy_model.tflite")
