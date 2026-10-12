import os
import numpy as np
import gradio as gr
import tensorflow as tf
from PIL import Image

# ============================================================
# KONFIGURASI
# ============================================================
MODEL_PATH = "best_model.keras"
IMG_SIZE = (224, 224)
CLASS_NAMES = ['Immature', 'Partially Ripe', 'Fully Ripe', 'OverRipe', 'Decayed']

CLASS_DESCRIPTIONS = {
    'Immature': 'Buah mentah, belum siap panen',
    'Partially Ripe': 'Setengah matang, masih perlu waktu',
    'Fully Ripe': 'Matang optimal, siap panen',
    'OverRipe': 'Terlalu matang, segera panen',
    'Decayed': 'Busuk, tidak layak panen'
}

# ============================================================
# LOAD MODEL
# ============================================================
print("Loading model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded: {MODEL_PATH}")
    print(f"Input shape : {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# ============================================================
# FUNGSI PREDIKSI
# ============================================================
def predict(image):
    """Prediksi kematangan buah sawit dari gambar."""
    if image is None:
        return {"Error": 1.0}, "Silakan upload gambar terlebih dahulu."

    # Preprocess
    img = Image.fromarray(image).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    proba = model.predict(img_array, verbose=0)[0]
    pred_idx = int(np.argmax(proba))
    pred_label = CLASS_NAMES[pred_idx]
    confidence = float(proba[pred_idx])

    # Probabilitas semua kelas
    result = {CLASS_NAMES[i]: float(proba[i]) for i in range(len(CLASS_NAMES))}

    # Info tambahan
    info = f"**Prediksi: {pred_label}** ({confidence*100:.1f}% confidence)\n\n"
    info += f"{CLASS_DESCRIPTIONS[pred_label]}\n\n"
    info += "**Probabilitas semua kelas:**\n"
    for label, prob in sorted(result.items(), key=lambda x: -x[1]):
        bar = "#" * int(prob * 30)
        info += f"- {label}: {prob*100:.1f}% {bar}\n"

    info += "\n**Catatan:** Confidence tinggi bukan jaminan prediksi benar. "
    info += "Model ini dilatih pada dataset penelitian dan belum diuji pada kondisi kebun sesungguhnya. "
    info += "Bukan pengganti penilaian ahli agronomi."

    return result, info

# ============================================================
# GRADIO INTERFACE
# ============================================================
title = "Smart Harvest Intelligence"
description = """
**Klasifikasi Kematangan Buah Sawit dengan AI**

Model MobileNetV2 yang dilatih pada 4.728 gambar buah sawit dari Central Kalimantan, Indonesia.

**5 Kelas Kematangan:**
- Immature (mentah)
- Partially Ripe (setengah matang)
- Fully Ripe (matang optimal)
- OverRipe (terlalu matang)
- Decayed (busuk)

**Cara pakai:** Upload foto buah sawit, model akan memprediksi tingkat kematangannya.
"""

article = """
### Tentang Model

- **Arsitektur:** MobileNetV2 (transfer learning)
- **Dataset:** 4.728 gambar, 5 kelas ordinal
- **Split:** 3.306 train / 704 validation / 718 test
- **F1 macro (test set):** 0.8079
- **Ordinal MAE:** 0.2521
- **Sumber dataset:** Mendeley Data (424y96m6sw)

### Keterbatasan

- Dilatih pada dataset penelitian, belum diuji pada kondisi kebun nyata
- Hanya klasifikasi gambar utuh, bukan deteksi posisi buah
- Confidence belum terkalibrasi
- Bukan pengganti penilaian ahli agronomi

### Info

Proyek: **Smart Harvest Intelligence** — Phase 2 Demo
"""

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Upload Foto Buah Sawit"),
    outputs=[
        gr.Label(label="Probabilitas Kelas", num_top_classes=5),
        gr.Markdown(label="Detail Prediksi")
    ],
    title=title,
    description=description,
    article=article
)

# ============================================================
# LAUNCH
# ============================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
