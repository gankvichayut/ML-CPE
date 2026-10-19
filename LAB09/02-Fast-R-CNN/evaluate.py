from pathlib import Path
import argparse
import json
import numpy as np
import cv2
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

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


def load_crops(split="test", limit=200):
    images_dir = DATASET / split / "images"
    labels_dir = DATASET / split / "labels"

    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f"Missing dataset folders: '{images_dir}' or '{labels_dir}'")

    files = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    if limit > 0:
        files = files[:limit]

    xs, ys = [], []

    for path in files:
        label = labels_dir / (path.stem + ".txt")
        if not label.exists():
            continue

        im = cv2.imread(str(path))
        if im is None:
            continue

        h, w = im.shape[:2]
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

            xs.append(preprocess(im[y1:y2, x1:x2]))
            ys.append(cls)

    if not xs:
        raise RuntimeError(f"No labeled crops found in {images_dir}.")

    return np.asarray(xs, np.float32), np.asarray(ys, np.int32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--batch", type=int, default=32)
    args = p.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}\nPlease run train.py first.")

    print(f"Loading '{args.split}' set for evaluation...")
    x, y = load_crops(args.split, args.limit)
    print(f"Total labeled crops evaluated: {len(y)}")

    model = tf.keras.models.load_model(MODEL_PATH)
    pred = model.predict(x, batch_size=args.batch, verbose=1).argmax(axis=1)

    acc = float(accuracy_score(y, pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, average="macro", labels=list(range(len(NAMES))), zero_division=0
    )

    metrics = {
        "split": args.split,
        "crops": len(y),
        "accuracy": round(acc, 4),
        "precision_macro": round(float(precision), 4),
        "recall_macro": round(float(recall), 4),
        "f1_macro": round(float(f1), 4)
    }

    print("\n" + "=" * 45)
    print(f"Evaluation Summary (Accuracy: {acc * 100:.2f}%)")
    print("=" * 45)
    print(json.dumps(metrics, indent=2))
    print("\nClassification Report:")
    print(classification_report(y, pred, labels=list(range(len(NAMES))), target_names=NAMES, zero_division=0))

    out_file = OUT / "evaluation.json"
    out_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved evaluation metrics to: {out_file}")


if __name__ == "__main__":
    main()