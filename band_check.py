import tensorflow as tf
import os

MODEL_PATH = "data/hyderabad/unet_hyderabad_multiclass.h5"

# Load model for inference only
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Check model summary
model.summary()
