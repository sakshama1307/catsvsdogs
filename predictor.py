import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# 1. Load your trained model
@st.cache_resource
def load_model():
    # Use dynamic pathing for safe deployment
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'best_model.keras')
    
    # compile=False prevents the Keras Adam optimizer deserialization error
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

model = load_model()

# 2. Image Preprocessing Function
# 2. Image Preprocessing Function
def preprocess_image(image):
    # Updated to 256x256 to match the model's InputLayer
    image = image.resize((256, 256))
    img_array = np.array(image)
    
    # Handle greyscale or RGBA images by enforcing 3 RGB channels
    if img_array.ndim == 2:
        img_array = np.stack((img_array,)*3, axis=-1)
    elif img_array.shape[2] == 4:
        img_array = img_array[:,:,:3]
        
    # REMOVED: img_array = img_array / 255.0 
    # (Your model handles rescaling internally!)
    
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# 3. Streamlit UI
st.title("🐱 vs 🐶 Cat or Dog Classifier")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    if st.button('Predict'):
        with st.spinner('Analyzing...'):
            processed_img = preprocess_image(image)
            prediction = model.predict(processed_img)
            
            # Binary classification: 0 for Cat, 1 for Dog (Adjust if your labels are flipped)
            if prediction[0][0] > 0.5:
                st.success("Prediction: **Dog**")
            else:
                st.success("Prediction: **Cat**")