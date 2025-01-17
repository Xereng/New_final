import streamlit as st
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import google.generativeai as genai

# Configure Generative AI
GEMINI_API_KEY = "AIzaSyB_vq5Y9ER24NsTCWShsgN-SAATR_zbPg0"
genai.configure(api_key=GEMINI_API_KEY)

# Load the trained model
MODEL_PATH = "mobilenetv2_fish_disease_model.h5"
try:
    model = load_model(MODEL_PATH)
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Model input dimensions and class names
IMG_HEIGHT = 150
IMG_WIDTH = 150
CLASS_NAMES = [
    "Bacterial diseases-Aeromoniasis",
    "Bacterial gill disease",
    "Bacterial red disease",
    "Fungal diseases-Saprolegniasis",
    "Healthy fish",
    "Parasitic diseases",
    "Viral diseases with tail disease"
]

def fetch_remedies(disease_name):
    """Fetch remedies for the predicted disease using Google Generative AI."""
    try:
        # Generate content using Generative AI
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(f"Provide remedies for {disease_name} in fish")
        return response.text
    except Exception as e:
        return f"Error fetching remedies: {e}"

# Streamlit UI
st.title("Fish Disease Detector using Mobile netv2 model")

st.write("""
    Upload an image of a fish to classify whether it is healthy or infected with a disease and get remedies.
""")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Show the uploaded image
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

    # Predict and fetch remedies when the "Classify" button is clicked
    if st.button("Classify"):
        try:
            # Process the uploaded image
            image = load_img(uploaded_file, target_size=(IMG_HEIGHT, IMG_WIDTH))
            image_array = img_to_array(image) / 255.0  # Normalize the image
            image_array = np.expand_dims(image_array, axis=0)

            # Predict the disease
            predictions = model.predict(image_array)
            predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
            confidence = np.max(predictions[0])

            st.subheader(f"Prediction: {predicted_class}")
            st.write(f"Confidence: {confidence * 100:.2f}%")

            # Fetch and display remedies
            if predicted_class != "Healthy fish":
                st.subheader("Remedies")
                remedies = fetch_remedies(predicted_class)
                st.write(remedies)
            else:
                st.write("The fish appears to be healthy. No remedies required.")

        except Exception as e:
            st.error(f"Error processing image: {e}")
