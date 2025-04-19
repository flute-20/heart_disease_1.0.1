import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import gdown

# Title of the app
st.title("Heart Disease Prediction from MRI Images")

# Upload the image
uploaded_file = st.file_uploader("Upload an MRI image (PNG/JPG)", type=["png", "jpg", "jpeg"])

# Download model from Google Drive
@st.cache_resource
def load_model():
    url = 'https://drive.google.com/uc?id=1H6PyS3lqIDA7HJYQF2fG8IvZiZP2zvcG'  # Your Google Drive file link
    output = 'model.keras'
    gdown.download(url, output, quiet=False)
    
    # Load the model
    model = tf.keras.models.load_model(output)
    return model

# Load the model once using Streamlit cache
model = load_model()

# Class labels (based on your training)
class_labels = ['Normal', 'Hypertrophy', 'Heart failure with infraction (HF-1)', 'Heart failure without infraction (HF)']

# Function to preprocess the image
def preprocess_image(image):
    # Convert BGR to RGB (because OpenCV loads in BGR format)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize to match model input size
    image = cv2.resize(image, (299, 299))
    # Normalize the image
    image = image / 255.0
    # Expand dimensions to match model input shape (1, 299, 299, 3)
    image = np.expand_dims(image, axis=0)
    return image

# Predict function
def predict(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][predicted_class] * 100
    return class_labels[predicted_class], confidence

# If an image is uploaded, process and predict
if uploaded_file is not None:
    # Read the image using OpenCV
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    if image is None:
        st.error("Error reading the image. Please upload a valid image file.")
        st.stop()
    
    # Display the uploaded image
    st.image(image, caption="Uploaded MRI Image", use_container_width=True)
    
    # Predict
    with st.spinner("Predicting..."):
        predicted_class, confidence = predict(image)
    
    # Display the result
    st.success(f"**Prediction:** {predicted_class} \n**Confidence:** {confidence:.2f}%")
else:
    st.info("Please upload an MRI image to get a prediction.")

# Footer
st.markdown("---")
st.write("Built with ❤ by Prendu using Streamlit and TensorFlow")
