"""
app.py — Bangla OCR  |  Streamlit Prediction UI
================================================
Run:
    streamlit run app.py

Requires:
    models/model.keras     — trained with train.py
    labels.json            — produced by train.py
    artifacts/mlflow/      — MLflow tracking directory (produced by train.py)
"""

import json
import os

import cv2
import numpy as np
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="বাংলা OCR",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS: Noto Serif Bengali + clean neutral palette ───────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+Bengali:wght@400;700&display=swap');

.bangla-word {
    font-family: 'Noto Serif Bengali', serif;
    font-size: 2.8rem;
    font-weight: 700;
    color: #111;
    margin: 0.4rem 0;
}
.bangla-char {
    font-family: 'Noto Serif Bengali', serif;
    font-size: 1.5rem;
    color: #111;
}
.section-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 0.3rem;
}
.conf-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.45rem;
}
.conf-track {
    flex: 1;
    height: 8px;
    background: #e5e5e5;
    border-radius: 4px;
    overflow: hidden;
}
.conf-fill { height: 100%; border-radius: 4px; }
.conf-pct  { font-size: 0.72rem; color: #888; min-width: 2.8rem; text-align: right; }

.mlflow-card {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    border-left: 4px solid #4f46e5;
    font-size: 0.78rem;
}
.mlflow-card b { color: #111; }
.mlflow-card span { color: #555; }

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Cached loaders
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading model…")
def load_model_and_labels(model_path: str, labels_path: str):
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path)
    with open(labels_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return model, data["index_to_char"]


@st.cache_data(show_spinner=False)
def load_mlflow_runs(mlflow_uri: str, experiment_name: str) -> list[dict]:
    """Return a list of dicts describing completed MLflow runs."""
    try:
        import mlflow
        mlflow.set_tracking_uri(mlflow_uri)
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(experiment_name)
        if exp is None:
            return []
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=20,
        )
        result = []
        for r in runs:
            m = r.data.metrics
            p = r.data.params
            result.append({
                "run_id":   r.info.run_id[:8],
                "run_name": r.info.run_name or "—",
                "status":   r.info.status,
                "phase":    p.get("phase", "—"),
                "epochs":   p.get("epochs") or p.get("fine_tune_epochs", "—"),
                "lr":       p.get("lr") or p.get("fine_tune_lr", "—"),
                "val_acc":  f'{m["val_accuracy"]:.4f}' if "val_accuracy" in m else "—",
                "test_acc": f'{m["test_accuracy"]:.4f}' if "test_accuracy" in m else "—",
                "top3_acc": f'{m["test_top3_acc"]:.4f}' if "test_top3_acc" in m else "—",
            })
        return result
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# Image helpers
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_canvas(rgba: np.ndarray) -> np.ndarray:
    """RGBA canvas → inverted binary (white strokes on black, matching training)."""
    gray = cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)
    gray = cv2.bitwise_not(gray)
    _, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    return binary


def find_segments(binary: np.ndarray,
                  min_width: int = 8,
                  merge_gap: int = 12) -> list[tuple[int, int, int, int]]:
    """
    Vertical-projection segmentation.

    Strategy:
    1. Sum pixel values along columns → projection profile.
    2. Identify contiguous non-zero column runs as raw segments.
    3. Merge runs whose gap ≤ merge_gap (handles ligatures / matras).
    4. Compute tight vertical bounds per segment + small padding.

    Returns list of (x1, y1, x2, y2) bounding boxes.
    """
    h, w = binary.shape
    proj = np.sum(binary, axis=0)

    # Step 1 & 2 — raw column runs
    raw: list[tuple[int, int]] = []
    in_seg, x0 = False, 0
    for x in range(w):
        if proj[x] > 0 and not in_seg:
            in_seg, x0 = True, x
        elif proj[x] == 0 and in_seg:
            in_seg = False
            if x - x0 >= min_width:
                raw.append((x0, x))
    if in_seg and w - x0 >= min_width:
        raw.append((x0, w))

    if not raw:
        return []

    # Step 3 — merge nearby runs
    merged = [list(raw[0])]
    for x1, x2 in raw[1:]:
        if x1 - merged[-1][1] <= merge_gap:
            merged[-1][1] = x2
        else:
            merged.append([x1, x2])

    # Step 4 — tight vertical bounds + padding
    PAD = 4
    boxes: list[tuple[int, int, int, int]] = []
    for x1, x2 in merged:
        strip = binary[:, x1:x2]
        rows  = np.any(strip > 0, axis=1)
        if not np.any(rows):
            continue
        y1 = max(0, int(np.argmax(rows)) - PAD)
        y2 = min(h, int(h - np.argmax(rows[::-1])) + PAD)
        boxes.append((max(0, x1 - PAD), y1, min(w, x2 + PAD), y2))
    return boxes


def crop_and_resize(binary: np.ndarray, box: tuple, size: int = 64) -> np.ndarray:
    """Crop → pad to square → resize → normalise to [0,1] RGB."""
    x1, y1, x2, y2 = box
    img  = binary[y1:y2, x1:x2]
    h, w = img.shape
    side  = max(h, w, 1)
    ph, pw = (side - h) // 2, (side - w) // 2
    img = cv2.copyMakeBorder(img, ph, side - h - ph, pw, side - w - pw,
                             cv2.BORDER_CONSTANT, value=0)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return (cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0)


def run_prediction(model, index_to_char: dict,
                   char_imgs: list[np.ndarray],
                   top_k: int = 5) -> list[dict]:
    """Batch inference → list of {char, confidence, topk}."""
    if not char_imgs:
        return []
    probs = model.predict(np.stack(char_imgs), verbose=0)
    results = []
    for p in probs:
        top = np.argsort(p)[::-1][:top_k]
        results.append({
            "char":       index_to_char[str(top[0])],
            "confidence": float(p[top[0]]),
            "topk":       [(index_to_char[str(i)], float(p[i])) for i in top],
        })
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Render helpers
# ═══════════════════════════════════════════════════════════════════════════════

def conf_color(c: float) -> str:
    if c >= 0.75: return "#16a34a"
    if c >= 0.45: return "#d97706"
    return "#dc2626"


def show_result_word(predictions: list[dict]):
    word     = "".join(p["char"] for p in predictions)
    avg_conf = float(np.mean([p["confidence"] for p in predictions]))
    st.markdown('<p class="section-label">Recognised word</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="bangla-word">{word}</p>', unsafe_allow_html=True)
    color = conf_color(avg_conf)
    st.markdown(
        f'<span style="font-size:0.8rem;color:{color};font-weight:600">'
        f'Avg confidence: {avg_conf:.0%}</span>',
        unsafe_allow_html=True,
    )
    st.divider()


def show_confidence_bars(predictions: list[dict]):
    st.markdown('<p class="section-label">Confidence per character</p>',
                unsafe_allow_html=True)
    for p in predictions:
        pct   = int(p["confidence"] * 100)
        color = conf_color(p["confidence"])
        st.markdown(f"""
        <div class="conf-row">
            <span class="bangla-char">{p['char']}</span>
            <div class="conf-track">
                <div class="conf-fill" style="width:{pct}%;background:{color}"></div>
            </div>
            <span class="conf-pct">{pct}%</span>
        </div>""", unsafe_allow_html=True)
    st.divider()


def show_topk_table(predictions: list[dict]):
    with st.expander("Top-5 candidates per character"):
        cols = st.columns(min(len(predictions), 8))
        for i, (pred, col) in enumerate(zip(predictions, cols)):
            with col:
                st.caption(f"Char {i + 1}")
                for rank, (ch, pr) in enumerate(pred["topk"]):
                    weight = "700" if rank == 0 else "400"
                    color  = "#111" if rank == 0 else "#888"
                    st.markdown(
                        f'<span class="bangla-char" style="font-weight:{weight};'
                        f'color:{color}">{ch}</span>'
                        f'<span style="font-size:0.7rem;color:#aaa"> {pr:.0%}</span><br>',
                        unsafe_allow_html=True,
                    )


def show_char_thumbnails(char_imgs: list[np.ndarray], predictions: list[dict]):
    st.markdown('<p class="section-label">Segmented characters</p>',
                unsafe_allow_html=True)
    cols = st.columns(min(len(char_imgs), 10))
    for i, (img_arr, pred) in enumerate(zip(char_imgs, predictions)):
        with cols[i % len(cols)]:
            st.image((img_arr * 255).astype(np.uint8), width=50)
            st.markdown(
                f'<div style="text-align:center" class="bangla-char">'
                f'{pred["char"]}</div>',
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar — config + MLflow run browser
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ Settings")
    model_path      = st.text_input("model.keras path",  value="models/model.keras")
    labels_path     = st.text_input("labels.json path",  value="labels.json")
    img_size        = st.selectbox("Model input size (px)", [32, 64, 96, 128], index=1)
    mlflow_uri      = st.text_input("MLflow tracking URI", value="./artifacts/mlflow")
    experiment_name = st.text_input("MLflow experiment",   value="bangla_ocr")

    st.subheader("Segmentation")
    merge_gap = st.slider("Merge gap (px)",         4, 30, 12,
                          help="Increase to merge over-split characters")
    min_w     = st.slider("Min segment width (px)", 4, 20,  8)

    st.caption("BanglaLekha-Isolated · 84 classes")

    # ── MLflow run browser ───────────────────────────────────────────────────
    st.divider()
    st.subheader("📊 MLflow Runs")
    if st.button("🔄 Refresh runs"):
        load_mlflow_runs.clear()

    runs = load_mlflow_runs(mlflow_uri, experiment_name)
    if not runs:
        st.caption("No runs found. Train first with `python train.py`.")
    else:
        for r in runs:
            status_icon = "✅" if r["status"] == "FINISHED" else "⏳"
            st.markdown(f"""
            <div class="mlflow-card">
                <b>{status_icon} {r['run_name']}</b><br>
                <span>ID: {r['run_id']} &nbsp;|&nbsp; Phase: {r['phase']}</span><br>
                <span>Val acc: <b>{r['val_acc']}</b> &nbsp;|&nbsp;
                      Test acc: <b>{r['test_acc']}</b> &nbsp;|&nbsp;
                      Top-3: <b>{r['top3_acc']}</b></span>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Model status
# ═══════════════════════════════════════════════════════════════════════════════

model_ready = os.path.exists(model_path) and os.path.exists(labels_path)

if model_ready:
    model, index_to_char = load_model_and_labels(model_path, labels_path)
    demo_mode = False
else:
    demo_mode = True


# ═══════════════════════════════════════════════════════════════════════════════
# Page header
# ═══════════════════════════════════════════════════════════════════════════════

st.title("✍️ বাংলা OCR")
st.caption("Handwritten Bangla word recognition · BanglaLekha-Isolated")

if demo_mode:
    st.warning(
        "**Model not found** — running in demo mode with random predictions.  \n"
        "Train first: `python train.py --data_dir data/BanglaLekha-Isolated`"
    )
else:
    st.success("Model loaded ✓", icon="✅")

st.divider()


# ═══════════════════════════════════════════════════════════════════════════════
# Main layout: canvas | results
# ═══════════════════════════════════════════════════════════════════════════════

left, right = st.columns([1.1, 1], gap="large")

with left:
    st.subheader("Draw a Bangla word")
    st.caption("Write left-to-right · single word · press Recognise when done")

    try:
        from streamlit_drawable_canvas import st_canvas
        stroke_width = st.slider("Stroke width", 4, 20, 10)
        canvas_result = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=stroke_width,
            stroke_color="#000000",
            background_color="#ffffff",
            height=220,
            width=560,
            drawing_mode="freedraw",
            key="canvas",
            display_toolbar=True,
        )
    except ImportError:
        st.error("Run: `pip install streamlit-drawable-canvas`")
        canvas_result = None

    col1, col2 = st.columns(2)
    predict_btn = col1.button("🔍 Recognise", use_container_width=True)
    clear_btn   = col2.button("🗑️ Clear",     use_container_width=True)
    if clear_btn:
        st.rerun()

with right:
    st.subheader("Result")

    if predict_btn and canvas_result is not None:
        img_data = canvas_result.image_data

        if img_data is None or np.sum(img_data[:, :, 3]) < 500:
            st.info("Canvas is empty — draw a Bangla word first.")
        else:
            with st.spinner("Segmenting & recognising…"):
                binary = preprocess_canvas(img_data.astype(np.uint8))
                boxes  = find_segments(binary, min_width=min_w, merge_gap=merge_gap)

                if not boxes:
                    st.warning("No characters found. Try drawing with thicker strokes.")
                else:
                    char_imgs = [crop_and_resize(binary, b, img_size) for b in boxes]

                    if demo_mode:
                        sample = ["ক","খ","গ","ঘ","ঙ","চ","ছ","জ","ঝ","ঞ"]
                        predictions = [
                            {
                                "char":       np.random.choice(sample),
                                "confidence": float(np.random.uniform(0.5, 0.99)),
                                "topk":       [(c, float(np.random.uniform(0.01, 0.5)))
                                               for c in np.random.choice(sample, 5, replace=False)],
                            }
                            for _ in char_imgs
                        ]
                    else:
                        predictions = run_prediction(model, index_to_char, char_imgs)

                    show_result_word(predictions)
                    show_confidence_bars(predictions)
                    show_topk_table(predictions)
                    show_char_thumbnails(char_imgs, predictions)
    else:
        st.caption("Results will appear here after you draw and press Recognise.")
