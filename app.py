import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2

try:
    model = tf.keras.models.load_model('asl_mnist_model.h5')
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# --- App Title and Description ---
st.title("🤟 Hand Sign Language Classifier")
st.write("Upload an image of a hand sign, and the model will predict which letter it represents.")

# --- Class Labels ---
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

# --- Image Upload Widget ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# --- Image Preprocessing Function ---
def preprocess_image(image):
    """
    Correctly processes an image into the format the model expects.
    """
    image_array = np.array(image)

    gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    resized_image = cv2.resize(gray_image, (28, 28))

    three_channel_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)


    reshaped_image = np.expand_dims(three_channel_image, axis=0)

    normalized_image = reshaped_image / 255.0
    
    return normalized_image

# --- Main App Logic ---
if uploaded_file is not None and model is not None:
    # Display the user's uploaded image.
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image to the correct format.
    processed_image = preprocess_image(image)

    # Make a prediction.
    prediction = model.predict(processed_image)
    
    # Get the top prediction's index and confidence score.
    predicted_class_index = np.argmax(prediction)
    confidence = np.max(prediction)
    
    if 0 <= predicted_class_index < len(class_labels):
        predicted_class_label = class_labels[predicted_class_index]
        # Display the final result.
        st.success(f"**Prediction:** {predicted_class_label}")
        st.write(f"**Confidence:** {confidence:.2f}")
    else:
        st.error(f"Prediction index ({predicted_class_index}) is out of range for the class labels.")