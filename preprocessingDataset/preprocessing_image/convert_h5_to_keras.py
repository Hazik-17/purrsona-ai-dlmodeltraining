import tensorflow as tf

# This small script reads a h5 model file and saves it in the newer keras format
# Input the h5 model file in this folder
# Output a new .keras model file saved next to this script
model = tf.keras.models.load_model('cat_breed_mobilenetv3_training_corrected.h5')
model.save('cat_breed_model_mobilenetv3.keras')
print('Successfully converted to .keras format')
