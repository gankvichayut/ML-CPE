import json
import os
import numpy as np

from data_loader import load_data
from preprocessing import to_features
from split_data import split_dataset
from cnn_model import train_model, predict_model
from evaluate import evaluate_model, plot_models_comparison

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "rps-cv-images")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

IMG_SIZE = 100
TEST_SIZE = 0.2
VAL_SIZE = 0.1
MAX_PER_CLASS = 1000   # ปรับเพื่อความเร็วในการรันหลายโมเดล หรือใส่ None เพื่อใช้ทั้งหมด
EPOCHS = 35
BATCH_SIZE = 64


def main():
    print("=" * 60)
    print("CNN Comparison Pipeline: Rock Paper Scissors")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. โหลดข้อมูล
    images, labels, classes = load_data(DATA_PATH, IMG_SIZE, MAX_PER_CLASS)
    with open(f"{OUTPUT_DIR}/classes.json", "w") as f:
        json.dump(classes, f)

    # 2. แบ่งข้อมูล
    X = to_features(images)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        X, labels, TEST_SIZE, VAL_SIZE
    )
    np.save(f"{OUTPUT_DIR}/X_test.npy", X_test)
    np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)

    # 3. กำหนด Configuration ที่จะเปรียบเทียบตามโจทย์ Lab
    # ปรับรายการ configurations ให้ความซับซ้อนต่างกันชัดเจน
    configurations = [
        {"name": "1-Conv Dense-32",  "conv_layers": 1, "dense": 32},
        {"name": "2-Conv Dense-64",  "conv_layers": 2, "dense": 64},
        {"name": "3-Conv Dense-128", "conv_layers": 3, "dense": 128},
        {"name": "4-Conv Dense-256", "conv_layers": 4, "dense": 256},
    ]

    histories = {}
    best_model = None
    best_acc = 0.0

    # 4. ลูปเทรนแต่ละ Configuration
    for cfg in configurations:
        print(f"\n>>> กำลังเทรน Model Configuration: {cfg['name']} ({EPOCHS} Epochs)")
        model, history = train_model(
            X_train, y_train, X_val, y_val,
            num_classes=len(classes),
            num_conv_layers=cfg["conv_layers"],
            dense_units=cfg["dense"],
            epochs=EPOCHS,
            batch_size=BATCH_SIZE
        )
        histories[cfg["name"]] = history

        # คำนวณ Accuracy เพื่อเลือกโมเดลที่ดีที่สุด
        preds = predict_model(model, X_test)
        acc = float((preds == y_test).mean())
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_model.save(f"{OUTPUT_DIR}/cnn_model.keras")

    # 5. สร้างกราฟเปรียบเทียบแบบเดียวกับรูปของอาจารย์
    plot_models_comparison(histories, f"{OUTPUT_DIR}/models_comparison.png")

    # 6. ประเมินโมเดลตัวที่ดีที่สุด
    best_preds = predict_model(best_model, X_test)
    evaluate_model(y_test, best_preds, classes, save_path=f"{OUTPUT_DIR}/confusion_matrix.png")


if __name__ == "__main__":
    main()