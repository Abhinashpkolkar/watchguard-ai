# Watchguard ai

Detects falls in a video using a Roboflow object-detection model
(`fall-detection-qyulz/2`). Includes a command-line script and a Streamlit app.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Provide your Roboflow API key (never commit it):

```bash
export ROBOFLOW_API_KEY="your_key"   # Windows PowerShell: $env:ROBOFLOW_API_KEY="your_key"
```

For the Streamlit app you can instead create `.streamlit/secrets.toml`
(copy from `.streamlit/secrets.toml.example`).

## Run

Command line (place a `falling_input.mp4` next to the script):

```bash
python fall_detection.py
```

Streamlit app:

```bash
streamlit run streamlit_app.py
```

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (the key stays out of the repo thanks to `.gitignore`).
2. Go to share.streamlit.io, pick the repo, set the main file to `streamlit_app.py`.
3. In the app's **Settings → Secrets**, add:
   ```
   ROBOFLOW_API_KEY = "your_key"
   ```
4. Deploy.

## Notes

- Inference runs one API call per frame, which is slow and consumes Roboflow
  quota. For long videos consider sampling every Nth frame or the `inference-sdk`.
- `requirements.txt` pins `opencv-python-headless` so it works on servers.
  The CLI is deliberately GUI-free (no `cv2.imshow`), so this one requirements
  file works everywhere. For a live preview, use the Streamlit app, which
  renders frames in the browser via `st.image` — no OpenCV GUI needed.
