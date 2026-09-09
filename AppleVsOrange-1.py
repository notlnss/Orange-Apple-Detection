import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# =========================================================
# KONFIGURASI (harus SAMA persis dengan waktu training)
# =========================================================
MODEL_PATH = "final_best_model.keras"
IMG_SIZE = 128
CLASS_NAMES = ["apple", "orange"]  # urutan sesuai sorted(os.listdir(DATASET_DIR))

st.set_page_config(page_title="Klasifikasi Buah", page_icon="🍎", layout="centered")


@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocessing HARUS sama persis dengan training:
    - resize ke (IMG_SIZE, IMG_SIZE)
    - rescale 1/255 (mobilenet_preprocess sudah dibungkus di dalam model,
      jadi JANGAN panggil manual lagi di sini)
    """
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)  # jadi (1, IMG_SIZE, IMG_SIZE, 3)
    return arr


def main():
    st.title("🍎🍊 Klasifikasi Buah: Apple vs Orange")
    st.write(
        "Upload gambar buah, model akan memprediksi apakah itu **apple** atau **orange** "
        "menggunakan MobileNetV2 Transfer Learning."
    )

    model = load_model()

    uploaded_file = st.file_uploader(
        "Upload gambar buah (jpg/png)", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diupload", use_container_width=True)

        with st.spinner("Memprediksi..."):
            input_arr = preprocess_image(image)
            pred_prob = model.predict(input_arr, verbose=0).ravel()[0]

        # Model output diasumsikan 1 neuron sigmoid (binary classification)
        pred_class_idx = int(pred_prob >= 0.5)
        pred_class = CLASS_NAMES[pred_class_idx]
        confidence = pred_prob if pred_class_idx == 1 else 1 - pred_prob

        st.subheader("Hasil Prediksi")
        st.success(f"Prediksi: **{pred_class.upper()}**")
        st.write(f"Confidence: **{confidence * 100:.2f}%**")

        st.write("---")
        st.write("Detail probabilitas:")
        st.write(f"- {CLASS_NAMES[0]}: {(1 - pred_prob) * 100:.2f}%")
        st.write(f"- {CLASS_NAMES[1]}: {pred_prob * 100:.2f}%")


if __name__ == "__main__":
    main()
