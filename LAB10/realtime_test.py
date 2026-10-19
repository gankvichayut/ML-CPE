from pathlib import Path
import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

def get_weights_path():
    candidates = [
        ROOT / "runs" / "detect" / "bdd100k" / "weights" / "best.pt",
        ROOT / "runs" / "bdd100k" / "weights" / "best.pt",
        ROOT / "runs" / "detect" / "runs" / "bdd100k" / "weights" / "best.pt"
    ]
    for p in candidates:
        if p.exists():
            return p
    found = list((ROOT / "runs").rglob("best.pt"))
    if found:
        return found[0]
    raise FileNotFoundError("Model not found. Please run train.py first.")

def main():
    weights = get_weights_path()
    model = YOLO(str(weights))
    device = 0 if torch.cuda.is_available() else "cpu"

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam (camera index 0)")

    print("Webcam started. Press 'Q' to quit real-time detection.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = model.predict(
                source=frame,
                conf=0.25,
                imgsz=480,
                device=device,
                verbose=False
            )

            annotated_frame = results[0].plot()
            cv2.imshow("BDD100K YOLO Real-time Detection (Q to exit)", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()