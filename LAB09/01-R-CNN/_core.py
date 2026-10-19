from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = Path(__file__).resolve().parent / "outputs" / "model.keras"
IMG_SIZE = 64

CLASSES = [
    "person", "rider", "car", "bus", "truck",
    "bike", "motor", "traffic light", "traffic sign", "train"
]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please run train.py first.")
    return tf.keras.models.load_model(MODEL_PATH)


def detect_demo(model, frame, threshold=0.80, n_regions=25):
    h, w = frame.shape[:2]
    preds = []
    crops = []
    coords = []

    # Simple sliding-window sampling
    side = min(h, w) // 3
    step_y = max(1, (h - side) // int(np.sqrt(n_regions)))
    step_x = max(1, (w - side) // int(np.sqrt(n_regions)))

    for y in range(0, h - side, step_y):
        for x in range(0, w - side, step_x):
            patch = frame[y:y + side, x:x + side]
            patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            patch_resized = cv2.resize(patch_rgb, (IMG_SIZE, IMG_SIZE))
            crops.append(patch_resized)
            coords.append((x, y, x + side, y + side))

    if not crops:
        return preds

    x_input = np.asarray(crops, dtype=np.float32) / 255.0
    scores = model.predict(x_input, verbose=0)

    for (x1, y1, x2, y2), score in zip(coords, scores):
        cls_idx = int(np.argmax(score))
        conf = float(np.max(score))
        if conf >= threshold:
            preds.append((x1, y1, x2, y2, CLASSES[cls_idx], conf))

    return preds


def draw(frame, preds):
    out = frame.copy()
    for x1, y1, x2, y2, label, conf in preds:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            out, f"{label} {conf*100:.1f}%",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )
    return out