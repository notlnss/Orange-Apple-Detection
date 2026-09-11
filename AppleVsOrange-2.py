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
LOW_CONFIDENCE_THRESHOLD = 0.60  # di bawah ini, tampilkan warning "mungkin bukan apple/orange"
MAX_FILE_SIZE_MB = 5

# Metrik dari notebook (hasil evaluate_model di test set)
MODEL_METRICS = {
    "Arsitektur": "MobileNetV2 (Transfer Learning)",
    "Accuracy": "0.9391",
    "F1 Score": "0.9391",
}

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


def predict(model, image: Image.Image):
    input_arr = preprocess_image(image)
    pred_prob = float(model.predict(input_arr, verbose=0).ravel()[0])
    pred_class_idx = int(pred_prob >= 0.5)
    pred_class = CLASS_NAMES[pred_class_idx]
    confidence = pred_prob if pred_class_idx == 1 else 1 - pred_prob
    return pred_class, confidence, pred_prob


def render_result(image: Image.Image, filename: str):
    st.image(image, caption=filename, width=280)

    with st.spinner("Memprediksi..."):
        pred_class, confidence, pred_prob = predict(load_model(), image)

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        st.warning(
            f"⚠️ Model kurang yakin (confidence {confidence * 100:.2f}%). "
            "Gambar mungkin bukan apple/orange, atau kualitas gambar kurang jelas. "
        )
    else:
        st.success(f"Prediksi: **{pred_class.upper()}**")

    st.write(f"Confidence: **{confidence * 100:.2f}%**")
    st.progress(int(confidence * 100))

    with st.expander("Detail probabilitas"):
        st.write(f"- {CLASS_NAMES[0]}: {(1 - pred_prob) * 100:.2f}%")
        st.progress(int((1 - pred_prob) * 100))
        st.write(f"- {CLASS_NAMES[1]}: {pred_prob * 100:.2f}%")
        st.progress(int(pred_prob * 100))


def main():
    with st.sidebar:
        st.header("ℹ️ Tentang Model")
        for key, val in MODEL_METRICS.items():
            st.write(f"**{key}:** {val}")
        st.write("---")
        st.caption(
            "⚠️ Model hanya mengenali 2 kelas: **apple** dan **orange**. "
            "Gambar buah lain atau objek non-buah tetap akan diklasifikasikan "
            "ke salah satu dari dua kelas ini."
        )

    st.title("🍎🍊 Klasifikasi Buah: Apple vs Orange")
    st.write(
        "Upload gambar buah (bisa lebih dari satu) "
        "**apple** atau **orange** menggunakan MobileNetV2 Transfer Learning."
    )

    tab_upload, tab_camera = st.tabs(["📁 Upload Gambar", "📷 Kamera"])

    with tab_upload:
        uploaded_files = st.file_uploader(
            "Upload gambar buah (jpg/png)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
                    st.error(f"❌ {uploaded_file.name} melebihi {MAX_FILE_SIZE_MB}MB, dilewati.")
                    continue

                st.subheader(uploaded_file.name)
                image = Image.open(uploaded_file)
                render_result(image, uploaded_file.name)
                st.write("---")

    with tab_camera:
        st.caption("Ambil foto langsung dari kamera (webcam di laptop, atau kamera HP).")
        camera_file = st.camera_input("Foto buah")

        if camera_file is not None:
            image = Image.open(camera_file)
            render_result(image, "Foto dari kamera")


if __name__ == "__main__":
    main()
