import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.models import load_model

model = tf.keras.models.load_model("digit_model.h5")

st.title("Handwritten Digit Recognition")

uploaded_file = st.file_uploader("Upload a digit image", type=["png","jpg","jpeg"])


if uploaded_file is not None:
    
    image = Image.open(uploaded_file).convert("L")

    st.image(image, caption="Uploaded Image", width=200)

    # resize to MNIST size
    image = image.resize((28,28))

    img_array = np.array(image)

    # normalize
    img_array = img_array / 255.0

    # reshape for CNN
    img_array = img_array.reshape(1,28,28,1)

    st.image(img_array.reshape(28,28), caption="Image seen by model", width=150)

    prediction = model.predict(img_array)

    digit = np.argmax(prediction)

    st.subheader(f"Predicted Digit: {digit}")