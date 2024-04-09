import streamlit as st
import tensorflow as tf  # Assuming TensorFlow is installed
import os

# Specific Model Path (update the filename if needed)
model_path = "C:/Users/suyash joshi/OneDrive/Desktop/Python Projects/Bell Pepper disease classifier/saved_models/bell pepper.h5"

# Load the model
try:
  model = tf.keras.models.load_model(model_path)
except FileNotFoundError as e:
  st.error(f"Error loading model: {e}")
  st.stop()  # Halt execution if model loading fails

# Title and description
st.title("Pepper Disease Prediction")
st.write("Upload an image of a pepper and predict if it has Bacterial Spot.")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# Button for prediction
predict_button = st.button("Predict")

if uploaded_file is not None and predict_button:
  # Read the image data
  image_bytes = uploaded_file.read()

  # Preprocess the image (replace with your specific preprocessing steps)
  # Example: Decode image and resize to expected input shape
  image = tf.image.decode_jpeg(image_bytes, channels=3)  # Assuming RGB image
  image = tf.image.resize(image, (224, 224))  # Example resize (adjust as needed)
  image = image / 255.0  # Normalize pixel values

  # Make prediction
  predictions = model(tf.expand_dims(image, axis=0))  # Add batch dimension
  predicted_class_index = tf.argmax(predictions, axis=1).numpy()[0]

  # Get class names (assuming these match your model's output order)
  class_names = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy']
  predicted_class = class_names[predicted_class_index]

  # Display the uploaded image
  st.image(image_bytes, caption="Uploaded Image")

  # Display the predicted class and class probabilities
  st.success(f"Predicted Class: {predicted_class}")
  st.write(f"Class Probabilities:")
  for i, class_name in enumerate(class_names):
    probability = predictions[0][i]
    st.write(f"- {class_name}: {probability:.2f}")

else:
  st.info("Upload an image and click 'Predict' to get a prediction.")
