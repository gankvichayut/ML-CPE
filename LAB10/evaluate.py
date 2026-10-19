from pathlib import Path
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

def get_weights_path():
    # ตรวจสอบตำแหน่งที่เป็นไปได้ของ best.pt
    candidates = [
        ROOT / "runs" / "detect" / "bdd100k" / "weights" / "best.pt",
        ROOT / "runs" / "bdd100k" / "weights" / "best.pt",
        ROOT / "runs" / "detect" / "runs" / "bdd100k" / "weights" / "best.pt"
    ]
    for p in candidates:
        if p.exists():
            return p
    # ถ้าไม่พบ ให้ลองค้นหาไฟล์ best.pt ใดๆ ในโฟลเดอร์ runs
    found = list((ROOT / "runs").rglob("best.pt"))
    if found:
        return found[0]
    raise FileNotFoundError(f"Model not found. Checked: {candidates[0]}\nPlease run train.py first.")

def main():
    weights = get_weights_path()
    print(f"Loading weights from: {weights}")

    model = YOLO(str(weights))
    device = 0 if torch.cuda.is_available() else "cpu"

    yaml_path = ROOT / "dataset" / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found at: {yaml_path}")

    metrics = model.val(
        data=str(yaml_path),
        split="test",
        imgsz=480,
        batch=8,
        device=device,
        workers=0,
        plots=True,
        project=str(ROOT / "runs"),
        name="evaluation",
        exist_ok=True,
    )

    print("\n===== Evaluation Results =====")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

if __name__ == "__main__":
    main()