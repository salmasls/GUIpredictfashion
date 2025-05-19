import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model

# Load model
model = load_model("fashion.h5")

# Label kelas Fashion MNIST
class_labels = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

st.title("Prediksi Fashion MNIST")
st.write("Unggah gambar item fashion (28x28 pixel grayscale)")

# Upload file gambar
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # Convert ke grayscale
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    # Pra-pemrosesan gambar
    image = ImageOps.invert(image)  # Fashion MNIST: putih di hitam
    image = image.resize((28, 28))
    img_array = np.array(image).astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Prediksi
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]

    st.write(f"**Prediksi:** {predicted_class}")
