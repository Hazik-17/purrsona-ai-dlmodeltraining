import tensorflow as tf

# === Step 1: Load the existing HDF5 model
model = tf.keras.models.load_model('cat_breed_mobilenetv3_training_85_05.h5')

# === Step 2: Save in new Keras format
model.save('cat_breed_model_mobilenetv3.keras')  # ✅ TF native format
print("✅ Successfully converted to .keras format")
