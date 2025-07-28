import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Load model
model = tf.keras.models.load_model('model_apel_baru.h5')

st.title("Klasifikasi Apel dan Kematangan")

uploaded_file = st.file_uploader("Upload gambar apel", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Konversi gambar ke RGB untuk memastikan hanya 3 channel
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Gambar yang di-upload', use_container_width=True)

    # Resize dan preprocessing gambar
    img = image.resize((150, 150))
    img_array = img_to_array(img) / 255.0
    img_tensor = tf.expand_dims(img_array, axis=0)  # Bentuk (1, 150, 150, 3)

    # Prediksi
    prediction = model.predict(img_tensor)
    is_apple = prediction[0][0][0]
    ripeness = np.argmax(prediction[1][0])

    ripeness_mapping = {
        0: "20% Matang",
        1: "40% Matang",
        2: "60% Matang",
        3: "80% Matang",
        4: "100% Matang"
    }

    # Tampilkan hasil
    st.subheader("Hasil Prediksi")
    st.write("Apel:", "✅ Ya" if is_apple > 0.5 else "❌ Bukan Apel")
    if is_apple > 0.5:
        st.write("Tingkat Kematangan:", ripeness_mapping[ripeness])
