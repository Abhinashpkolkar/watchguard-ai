import os
import tempfile

import cv2
import requests
import streamlit as st

# --- Config ---
PROJECT_ID = "fall-detection-qyulz"
MODEL_VERSION = "2"


def get_api_key():
    """Prefer Streamlit secrets, fall back to an environment variable."""
    try:
        return st.secrets["ROBOFLOW_API_KEY"]
    except Exception:
        return os.environ.get("ROBOFLOW_API_KEY")


def infer_frame(frame, api_key):
    url = (
        f"https://serverless.roboflow.com/{PROJECT_ID}/{MODEL_VERSION}"
        f"?api_key={api_key}"
    )
    _, img_encoded = cv2.imencode(".jpg", frame)
    resp = requests.post(url, files={"file": img_encoded.tobytes()})
    if resp.status_code == 200:
        return resp.json()
    st.warning(f"Inference error {resp.status_code}: {resp.text[:200]}")
    return None


def annotate(frame, predictions, threshold):
    if not predictions or "predictions" not in predictions:
        return frame
    for pred in predictions["predictions"]:
        conf = pred["confidence"]
        if conf < threshold:
            continue
        x, y = int(pred["x"]), int(pred["y"])
        w, h = int(pred["width"]), int(pred["height"])
        label = f"{pred['class'].lower()} ({conf:.2f})"
        color = (255, 255, 0)
        cv2.rectangle(
            frame, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), color, 3
        )
        cv2.putText(
            frame, label, (x - w // 2, y - h // 2 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
        )
    return frame


# --- UI ---
st.set_page_config(page_title="Fall Detection", layout="wide")
st.title("Fall Detection")
st.caption("Roboflow-powered fall detection on uploaded video.")

api_key = get_api_key()
if not api_key:
    st.error(
        "No Roboflow API key found. Add ROBOFLOW_API_KEY to your Streamlit "
        "secrets (Settings → Secrets) or set it as an environment variable."
    )
    st.stop()

uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.2, 0.05)

if uploaded is not None and st.button("Run detection"):
    # Save the upload to a temp file so OpenCV can read it.
    in_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    in_file.write(uploaded.read())
    in_file.close()

    cap = cv2.VideoCapture(in_file.name)
    if not cap.isOpened():
        st.error("Could not open the uploaded video.")
        st.stop()

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    frame_slot = st.empty()
    progress = st.progress(0.0)
    status = st.empty()

    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        i += 1
        preds = infer_frame(frame, api_key)
        frame = annotate(frame, preds, threshold)
        writer.write(frame)

        # Live preview (OpenCV is BGR, Streamlit expects RGB).
        frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        if total:
            progress.progress(min(i / total, 1.0))
        status.text(f"Processed {i}" + (f" / {total}" if total else "") + " frames")

    cap.release()
    writer.release()
    progress.progress(1.0)
    status.success(f"Done. {i} frames processed.")

    with open(out_path, "rb") as f:
        st.download_button(
            "Download annotated video",
            f,
            file_name="fall_output.mp4",
            mime="video/mp4",
        )
