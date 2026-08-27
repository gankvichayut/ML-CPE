"""
evaluate.py
หน้าที่:
1. พลอตกราฟเปรียบเทียบค่า k กับ Accuracy
2. สร้างและบันทึกภาพ Confusion Matrix
3. แสดงผล Classification Report (Precision, Recall, F1-Score)
4. บันทึกผลการทำนายจริงลงไฟล์ CSV
"""

import matplotlib
matplotlib.use("Agg")  # ปิด GUI Display เหมาะกับการรันบนสคริปต์/เซิร์ฟเวอร์

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

def plot_k_curve(k_values, scores, out_path):
    """วาดกราฟเส้นแสดงแนวโน้มความแม่นยำตามค่า k ต่าง ๆ"""
    plt.figure(figsize=(7, 4.5))
    plt.plot(k_values, scores, "o-", color="#1f77b4", linewidth=2, markersize=6)
    plt.title("KNN Hyperparameter Tuning (k vs Validation Accuracy)")
    plt.xlabel("k (Number of Neighbors)")
    plt.ylabel("Validation Accuracy")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, out_path):
    """วาด Confusion Matrix เพื่อดูข้อผิดพลาดในการทำนายแต่ละคลาส"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(len(class_names)), class_names, rotation=20)
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    # ใส่ตัวเลขจำนวนเคสลงในตาราง
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=12)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    return cm

def print_report(y_true, y_pred, class_names):
    """แสดงค่า Precision, Recall และ F1-Score ของแต่ละคลาส"""
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

def save_predictions(y_true, y_pred, class_names, out_path):
    """บันทึกผลการทำนายเทียบกับค่าจริงลงไฟล์ CSV"""
    df = pd.DataFrame({
        "true_label": [class_names[i] for i in y_true],
        "predicted_label": [class_names[i] for i in y_pred],
        "correct": y_true == y_pred,
    })
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return df