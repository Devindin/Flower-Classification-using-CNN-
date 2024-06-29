import os
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model

st.header('Flower Classification CNN Model')
flower_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

model = tf.keras.models.load_model('Flower_recognition.keras')

def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    outcome = (
        f"The Image belongs to {flower_names[np.argmax(result)]} "
        f"with a score of {np.max(result) * 100:.2f}%"
    )
    return outcome

uploaded_file = st.file_uploader('Upload an Image')
if uploaded_file is not None:
    upload_dir = 'upload'
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    elif os.path.isfile(upload_dir):
        os.remove(upload_dir)
        os.makedirs(upload_dir)
    
    file_path = os.path.join(upload_dir, uploaded_file.name)
    
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    st.image(uploaded_file, width=200)
    result = classify_images(file_path)
    st.markdown(result)
