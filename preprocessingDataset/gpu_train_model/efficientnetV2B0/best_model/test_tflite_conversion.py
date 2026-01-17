import tensorflow as tf

# Create a tiny dummy model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    tf.keras.layers.Conv2D(16, 3, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.save("dummy_model.keras")

print("✅ Dummy model saved. Now converting to TFLite...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("dummy_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Dummy TFLite model saved successfully!")
