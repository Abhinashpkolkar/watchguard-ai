# Importing libraries.
import os
import cv2
import requests

# Configurations.
# API key is read from the environment, never hardcoded.
# Set it before running, e.g.:  export ROBOFLOW_API_KEY="your_key"
API_KEY = os.environ.get('ROBOFLOW_API_KEY')
PROJECT_ID = 'fall-detection-qyulz'
MODEL_VERSION = '2'

# Inference URL.
# Correct Roboflow serverless format: {host}/{model}/{version}?api_key=KEY
INFERENCE_URL = (
    f'https://serverless.roboflow.com/{PROJECT_ID}/{MODEL_VERSION}'
    f'?api_key={API_KEY}'
)

# Input and output video paths.
INPUT_VIDEO_PATH = 'falling_input.mp4'
OUTPUT_VIDEO_PATH = 'fall_output_video.mp4'

# Confidence threshold.
CONFIDENCE_THRESHOLD = 0.2

# NOTE: This CLI script is intentionally GUI-free. It does NOT call
# cv2.imshow / cv2.waitKey, so it runs safely against opencv-python-headless
# (the build pinned in requirements.txt) on both servers and local machines.
# For a live, in-browser preview, use streamlit_app.py instead.


# Function to run inference on a single frame.
def infer_frame(frame):
    _, img_encoded = cv2.imencode('.jpg', frame)

    response = requests.post(
        INFERENCE_URL,
        files={'file': img_encoded.tobytes()},
    )

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None


# Main process.
def main():
    if not API_KEY:
        print("Error: ROBOFLOW_API_KEY environment variable is not set.")
        return

    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: could not open input video '{INPUT_VIDEO_PATH}'")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        predictions = infer_frame(frame)

        drawn = 0
        if predictions and 'predictions' in predictions:
            for pred in predictions['predictions']:
                confidence = pred['confidence']
                if confidence < CONFIDENCE_THRESHOLD:
                    continue

                x, y = int(pred['x']), int(pred['y'])
                w, h = int(pred['width']), int(pred['height'])
                display_label = pred['class'].lower()
                color = (255, 255, 0)

                cv2.rectangle(
                    frame,
                    (x - w // 2, y - h // 2),
                    (x + w // 2, y + h // 2),
                    color, 5,
                )
                label = f"{display_label}({confidence:.2f})"
                cv2.putText(
                    frame, label,
                    (x - w // 2, y - h // 2 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 5,
                )
                drawn += 1

        print(f"Processing frame {frame_count} - {drawn} detection(s) drawn")
        out.write(frame)

    cap.release()
    out.release()
    print(f"Output video saved to {OUTPUT_VIDEO_PATH}")


if __name__ == "__main__":
    main()
