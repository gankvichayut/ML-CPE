from pathlib import Path
import random
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
    raise FileNotFoundError(f"Model not found. Please run train.py first.")

def main():
    weights = get_weights_path()
    image_dir = ROOT / "dataset" / "test" / "images"

    if not image_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {image_dir}")

    model = YOLO(str(weights))
    device = 0 if torch.cuda.is_available() else "cpu"

    extensions = {".jpg", ".jpeg", ".png"}
    images = [p for p in image_dir.iterdir() if p.suffix.lower() in extensions]

    if not images:
        raise RuntimeError("No test images found in dataset/test/images")

    selected_images = random.sample(images, min(5, len(images)))

    for image in selected_images:
        results = model.predict(
            source=str(image),
            conf=0.25,
            imgsz=480,
            device=device,
            save=True,
            project=str(ROOT / "runs"),
            name="image_test",
            exist_ok=True
        )

        for result in results:
            print(f"\nImage: {image.name}")
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                print(f"  {model.names[class_id]}: {confidence:.2f}")

            annotated = result.plot()
            cv2.imshow("YOLO Image Detection (Press Any Key for Next, Q to Quit)", annotated)
            if cv2.waitKey(0) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print("\nImage testing completed! Annotated images saved to runs/image_test")

if __name__ == "__main__":
    main()