from pathlib import Path
import argparse
import random
import numpy as np
import cv2
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
MODEL_PATH = OUT / "model.keras"

NAMES = [
    "person", "rider", "car", "bus", "truck",
    "bike", "motor", "traffic light", "traffic sign", "train"
]
IMG_SIZE = 64


def preprocess(crop):
    resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", default=None)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    args = p.parse_args()

    images_dir = DATASET / args.split / "images"
    labels_dir = DATASET / args.split / "labels"

    if args.image:
        path = Path(args.image)
    else:
        if not images_dir.is_dir():
            raise FileNotFoundError(f"Missing images directory: {images_dir}")

        files = [
            p for p in images_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
            and (labels_dir / (p.stem + ".txt")).exists()
        ]

        if not files:
            raise RuntimeError(f"No labeled images found in {images_dir}")

        path = random.choice(files)

    frame = cv2.imread(str(path))
    if frame is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    label = labels_dir / (path.stem + ".txt")
    if not label.exists():
        raise FileNotFoundError(f"Ground-truth YOLO label required: {label}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}\nPlease run train.py first.")

    h, w = frame.shape[:2]
    crops = []
    boxes = []

    for line in label.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue

        try:
            cls = int(parts[0])
            xc, yc, bw, bh = map(float, parts[1:])
        except ValueError:
            continue

        if not 0 <= cls < len(NAMES):
            continue

        x1 = max(0, int((xc - bw / 2) * w))
        y1 = max(0, int((yc - bh / 2) * h))
        x2 = min(w, int((xc + bw / 2) * w))
        y2 = min(h, int((yc + bh / 2) * h))

        if x2 <= x1 or y2 <= y1:
            continue

        crops.append(preprocess(frame[y1:y2, x1:x2]))
        boxes.append((x1, y1, x2, y2))

    if not crops:
        raise RuntimeError("No valid labeled objects in image.")

    model = tf.keras.models.load_model(MODEL_PATH)
    predictions = model.predict(np.asarray(crops, np.float32), verbose=0)

    for (x1, y1, x2, y2), pred in zip(boxes, predictions):
        cls = int(np.argmax(pred))
        score = float(np.max(pred))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{NAMES[cls]} {score*100:.1f}%",
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (0, 255, 0), 2
        )

    target = OUT / "image_result.jpg"
    cv2.imwrite(str(target), frame)
    print(f"Sample Input Image : {path.name}")
    print(f"Objects Detected   : {len(boxes)}")
    print(f"Output Saved to    : {target}")


if __name__ == "__main__":
    main()