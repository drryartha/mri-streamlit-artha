import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageFilter
import os, json

# =========================
# CONFIG
# =========================
# ✅ GANTI PATH INI sesuai file terbaru kamu (MobileNetV3Large)
MODEL_PATH  = "model/cnn_brain_tumor.keras"
CLASS_PATH  = "model/class_names.json"
CONFIG_PATH = "model/config.json"

# Heuristic OOD gate (MRI) - saringan kasar
EDGE_DENSITY_MAX = 0.30
CONTRAST_MIN = 0.03
CONTRAST_MAX = 0.35

st.set_page_config(page_title="Brain Tumor MRI Classification", page_icon="🧠", layout="wide")

# =========================
# STYLE
# =========================
st.markdown(
    """
<style>
  .title {font-size: 44px; font-weight: 900; margin: 0.2rem 0 0.2rem 0;}
  .sub {font-size: 15px; opacity: 0.9; margin-top: 0;}
  .card {padding: 16px; border-radius: 16px; border: 1px solid rgba(255,255,255,0.10); background: rgba(255,255,255,0.03);}
  .pill {display:inline-block; padding: 6px 12px; border-radius: 999px; border: 1px solid rgba(255,255,255,0.18); background: rgba(255,255,255,0.05);}
  .hr {height:1px; background: rgba(255,255,255,0.12); margin: 12px 0;}
  .muted {opacity: 0.85;}
</style>
""",
    unsafe_allow_html=True
)

# =========================
# PREPROCESS (MobileNetV3)
# =========================
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mnv3_preprocess

# =========================
# LOAD MODEL + META
# =========================
@st.cache_resource
def load_assets():
    for p in [MODEL_PATH, CLASS_PATH, CONFIG_PATH]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"File not found: {p}")

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_PATH, "r") as f:
        class_names = json.load(f)

    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)

    img_size = tuple(cfg["img_size"])
    num_classes = len(class_names)

    # --- sanity checks (anti mismatch) ---
    if "num_classes" in cfg and int(cfg["num_classes"]) != num_classes:
        raise ValueError(f"num_classes mismatch: cfg={cfg['num_classes']} vs class_names={num_classes}")

    out_units = int(model.output_shape[-1])
    if out_units != num_classes:
        raise ValueError(f"Model output units ({out_units}) != len(class_names) ({num_classes})")

    # Warm-up (hindari error "never been called")
    dummy = tf.zeros((1, img_size[0], img_size[1], 3), dtype=tf.float32)
    _ = model(dummy, training=False)

    return model, class_names, img_size, num_classes, cfg

model, CLASS_NAMES, IMG_SIZE, NUM_CLASSES, CFG = load_assets()

# =========================
# PREPROCESS
# =========================
def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype(np.float32)

    # ✅ WAJIB: sama seperti training MobileNetV3Large
    arr = mnv3_preprocess(arr)

    return np.expand_dims(arr, axis=0)

# =========================
# INPUT GATE (HEURISTIC)
# =========================
def input_gate_metrics(img: Image.Image):
    """
    Heuristic check to reduce misleading predictions on non-MRI images.
    Returns: (is_ok, edge_density, contrast)
    """
    im = img.convert("RGB").resize((256, 256))
    arr = np.array(im).astype(np.float32) / 255.0

    edges = im.convert("L").filter(ImageFilter.FIND_EDGES)
    e = np.array(edges).astype(np.float32) / 255.0
    edge_density = float((e > 0.25).mean())

    contrast = float(arr.std(axis=(0, 1)).mean())

    is_ok = (edge_density <= EDGE_DENSITY_MAX) and (CONTRAST_MIN <= contrast <= CONTRAST_MAX)
    return is_ok, edge_density, contrast

# =========================
# MULTI-CROP INFERENCE (SOFTMAX)
# =========================
def get_five_crop_boxes(W: int, H: int, scale: float = 0.85):
    s = int(min(W, H) * scale)
    s = max(64, s)
    cx, cy = W // 2, H // 2
    boxes = [
        (cx - s//2, cy - s//2, cx + s//2, cy + s//2),  # center
        (0, 0, s, s),                                   # top-left
        (W - s, 0, W, s),                               # top-right
        (0, H - s, s, H),                               # bottom-left
        (W - s, H - s, W, H)                            # bottom-right
    ]
    out = []
    for (x0, y0, x1, y1) in boxes:
        out.append((max(0, x0), max(0, y0), min(W, x1), min(H, y1)))
    return out

def predict_multi_crop_softmax(model, img: Image.Image, crop_scale: float = 0.85):
    """
    Returns:
      pred_idx, probs_mean, conf_final,
      probs_per_crop, boxes, agreement, best_crop_index, top2
    """
    img = img.convert("RGB")
    W, H = img.size
    boxes = get_five_crop_boxes(W, H, scale=crop_scale)

    probs_per_crop = []
    pred_per_crop = []
    maxprob_per_crop = []

    for b in boxes:
        crop = img.crop(b)
        x = preprocess_image(crop)
        p = np.array(model.predict(x, verbose=0)[0], dtype=np.float32)  # softmax (C,)
        probs_per_crop.append(p)
        pred_per_crop.append(int(np.argmax(p)))
        maxprob_per_crop.append(float(np.max(p)))

    probs_per_crop = np.stack(probs_per_crop, axis=0)  # (K,C)
    probs_mean = probs_per_crop.mean(axis=0)           # (C,)

    pred_idx = int(np.argmax(probs_mean))
    chosen_prob = float(probs_mean[pred_idx])

    agreement = float(np.mean([1.0 if i == pred_idx else 0.0 for i in pred_per_crop]))

    spread = float(np.max(maxprob_per_crop) - np.min(maxprob_per_crop))
    stability = 1.0 - min(1.0, spread * 1.25)

    conf_final = float(chosen_prob * (0.55 + 0.25*agreement + 0.20*stability))

    best_crop_index = int(np.argmax(probs_per_crop[:, pred_idx]))

    top2_idx = probs_mean.argsort()[-2:][::-1]
    top2 = [(int(i), float(probs_mean[i])) for i in top2_idx]

    return pred_idx, probs_mean, conf_final, probs_per_crop, boxes, agreement, best_crop_index, top2

# =========================
# CONFIDENCE LABEL
# =========================
def confidence_level(conf: float, agreement: float):
    if conf >= 0.80 and agreement >= 0.80:
        return "High", "Stable prediction across crops."
    elif conf >= 0.60:
        return "Medium", "Moderate confidence; some crop disagreement."
    else:
        return "Low", "Unstable / ambiguous—verify carefully."

def render_confidence_badge(conf: float, agreement: float):
    level, desc = confidence_level(conf, agreement)
    if level == "High":
        st.success(f"Confidence: **{level}** — {desc}")
    elif level == "Medium":
        st.warning(f"Confidence: **{level}** — {desc}")
    else:
        st.error(f"Confidence: **{level}** — {desc}")

# =========================
# APP HEADER
# =========================
st.markdown('<div class="title">🧠 Brain Tumor MRI Classification</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub muted">Upload MRI otak untuk klasifikasi: <b>glioma</b>, <b>meningioma</b>, <b>pituitary</b>, atau <b>notumor</b>.</p>',
    unsafe_allow_html=True
)

# =========================
# SIDEBAR (tanpa explainable)
# =========================
with st.sidebar:
    st.markdown("## Panduan Singkat")
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    st.write("1) Upload gambar MRI otak (jpg/png).")
    st.write("2) Lihat hasil prediksi & tingkat keyakinan.")

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown("## Tips Upload")
    st.write("• Gunakan citra MRI otak (bukan foto orang/objek umum).")
    st.write("• Usahakan gambar jelas, tidak terlalu blur/gelap.")
    st.write("• Jika hasil tidak masuk akal, coba gambar MRI lain (kontras dataset bisa berbeda).")

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.caption("Disclaimer: Aplikasi ini bukan alat diagnosis klinis.")

# =========================
# TABS (tanpa Explain)
# =========================
tab_pred, tab_model, tab_about = st.tabs(["🔮 Predict", "📌 Model", "ℹ️ About"])

if "last_result" not in st.session_state:
    st.session_state.last_result = None

# =========================
# TAB: PREDICT
# =========================
with tab_pred:
    left, right = st.columns([1.1, 1.0], gap="large")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload image (jpg/png/jpeg)", type=["jpg", "jpeg", "png"])
        st.markdown("</div>", unsafe_allow_html=True)

        img = Image.open(uploaded) if uploaded is not None else None
        if img is not None:
            st.image(img, caption="Uploaded image", use_column_width=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Prediction")

        if img is None:
            st.info("Upload an image to see the prediction.")
            st.session_state.last_result = None
        else:
            ok, edge_density, contrast = input_gate_metrics(img)
            st.caption(f"Input check — edge_density: `{edge_density:.3f}` | contrast: `{contrast:.3f}`")

            if not ok:
                st.error("Gambar ini terlihat tidak seperti MRI otak (heuristic gate). Silakan upload MRI otak.")
                st.session_state.last_result = None
            else:
                pred_idx, probs_mean, conf, probs_per_crop, boxes, agreement, best_i, top2 = predict_multi_crop_softmax(
                    model, img, crop_scale=0.85
                )

                pred_label = CLASS_NAMES[pred_idx]
                st.markdown(f'<span class="pill">Prediction: <b>{pred_label}</b></span>', unsafe_allow_html=True)
                st.write("")

                render_confidence_badge(conf, agreement)
                st.caption(f"Crop agreement: **{agreement*100:.1f}%**")

                st.markdown("**Top-2 classes (mean prob):**")
                st.write(f"1) **{CLASS_NAMES[top2[0][0]]}** — `{top2[0][1]:.4f}`")
                st.write(f"2) **{CLASS_NAMES[top2[1][0]]}** — `{top2[1][1]:.4f}`")

                st.markdown("**All class probabilities (mean):**")
                st.json({CLASS_NAMES[i]: float(probs_mean[i]) for i in range(NUM_CLASSES)})

                st.write("Confidence:")
                st.progress(int(conf * 100))
                st.caption(f"{conf*100:.1f}%")

                st.write(f"Per-crop probability for predicted class (**{pred_label}**):")
                per_crop_pred = [float(p[pred_idx]) for p in probs_per_crop]
                st.bar_chart({"p_predicted_class_per_crop": per_crop_pred})

                st.session_state.last_result = {
                    "img": img,
                    "pred_idx": pred_idx,
                    "probs_mean": probs_mean,
                    "conf": conf,
                    "agreement": agreement,
                    "boxes": boxes,
                    "best_i": best_i,
                    "top2": top2
                }

        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# TAB: MODEL
# =========================
with tab_model:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Model details")
    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    st.code("\n".join(summary_lines), language="text")
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# TAB: ABOUT
# =========================
with tab_about:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("About")
    st.write(
        """
Aplikasi ini mengklasifikasikan citra MRI otak menjadi 4 kelas:
- glioma
- meningioma
- notumor
- pituitary

Fitur:
- Multi-crop inference (lebih stabil)
- Confidence dari agreement + stability
"""
    )
    st.markdown("#### Disclaimer")
    st.write("Ini demo software dan **bukan** alat diagnosis klinis.")
    st.markdown("</div>", unsafe_allow_html=True)

st.caption("© Brain Tumor MRI Classification")

