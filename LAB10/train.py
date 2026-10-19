from pathlib import Path
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

def main():
    # ตรวจสอบการใช้งาน GPU หรือ CPU อัตโนมัติ
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # โหลด pretrained YOLOv8n
    model = YOLO("yolov8n.pt")

    # ฝึกสอนโมเดล
    yaml_path = ROOT / "dataset" / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found at: {yaml_path}")

    model.train(
        data=str(yaml_path),
        epochs=10,
        imgsz=480,
        batch=32,
        device=device,
        workers=8,
        amp=torch.cuda.is_available(),
        cache=False,
        project=str(ROOT / "runs"),
        name="bdd100k",
        exist_ok=True,
        save=True,
        plots=True
    )

    save_dir = Path(model.trainer.save_dir)
    print("\nTraining completed!")
    print(f"Best Model: {save_dir / 'weights' / 'best.pt'}")
    print(f"Last Model: {save_dir / 'weights' / 'last.pt'}")
    print(f"Training CSV: {save_dir / 'results.csv'}")

if __name__ == "__main__":
    main()