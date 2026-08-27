"""
main.py
Pipeline การทำงาน:
- Step 1: โหลดและตรวจสอบขนาดชุดข้อมูล
- Step 2: ค้นหาค่า Best k จาก Validation Set
- Step 3: นำ Best k มาเทรนและประเมินผลบน Test Set
- Step 4: Cross-check ความถูกต้องของโค้ด TensorFlow เทียบกับ Scikit-Learn
- Step 5: ตรวจสอบประสิทธิภาพโมเดลเทียบกับการสุ่มเดา (Majority Class Baseline)
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # ซ่อน Log แจ้งเตือนที่ไม่จำเป็นของ TensorFlow

from pathlib import Path
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

import data_loader
import evaluate
from knn_tf import TFKNNClassifier

OUT_DIR = Path(__file__).resolve().parent / "outputs"

def title(text):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)

def main():
    OUT_DIR.mkdir(exist_ok=True)

    # -----------------------------------------------------------------------
    # STEP 1 : Load Data
    # -----------------------------------------------------------------------
    title("STEP 1 : Load and Prepare Dataset")
    data = data_loader.load_data()

    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    class_names = data["class_names"]

    print(f"Total instances   : {data['n_rows']} rows")
    print(f"Features count    : {X_train.shape[1]} features")
    print(f"Target classes    : {class_names}")
    print(f"Data split ratio  : Train {len(y_train)} | Val {len(y_val)} | Test {len(y_test)}")

    # -----------------------------------------------------------------------
    # STEP 2 : Hyperparameter Tuning (Find Best k)
    # -----------------------------------------------------------------------
    title("STEP 2 : Hyperparameter Tuning for Best k")
    k_values = [1, 3, 5, 7, 9, 11, 15, 21]
    scores = []

    for k in k_values:
        model = TFKNNClassifier(k=k)
        model.fit(X_train, y_train)
        acc = model.score(X_val, y_val)
        scores.append(acc)
        print(f"   k = {k:>2}  -->  Validation Accuracy = {acc:.4f}")

    best_k = k_values[int(np.argmax(scores))]
    print(f"\n[Result] Best k chosen from Validation: k = {best_k}")
    evaluate.plot_k_curve(k_values, scores, OUT_DIR / "01_k_curve.png")

    # -----------------------------------------------------------------------
    # STEP 3 : Train with Best k and Evaluate on Test Set
    # -----------------------------------------------------------------------
    title(f"STEP 3 : Final Evaluation on Test Set (k = {best_k})")
    model = TFKNNClassifier(k=best_k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = float(np.mean(y_pred == y_test))
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n")
    print("Class-wise Classification Report:")
    evaluate.print_report(y_test, y_pred, class_names)

    cm = evaluate.plot_confusion_matrix(y_test, y_pred, class_names, OUT_DIR / "02_confusion_matrix.png")
    print("Confusion Matrix:")
    print(cm)

    # -----------------------------------------------------------------------
    # STEP 4 : Compare with Scikit-learn Baseline
    # -----------------------------------------------------------------------
    title("STEP 4 : Verify Results with Scikit-learn")
    sk_model = KNeighborsClassifier(n_neighbors=best_k)
    sk_model.fit(X_train, y_train)
    sk_pred = sk_model.predict(X_test)
    sk_acc = float(np.mean(sk_pred == y_test))

    print(f"TensorFlow KNN Accuracy : {accuracy:.4f}")
    print(f"Scikit-Learn Accuracy   : {sk_acc:.4f}")
    print(f"Prediction Match Rate   : {np.mean(sk_pred == y_pred) * 100:.2f}%")

    # -----------------------------------------------------------------------
    # STEP 5 : Compare with Zero-Rule Baseline (Majority Guessing)
    # -----------------------------------------------------------------------
    title("STEP 5 : Performance vs Majority Class Guessing")
    majority = np.bincount(y_train).argmax()
    baseline = float(np.mean(y_test == majority))

    print(f"Majority Baseline Accuracy : {baseline:.4f} (Predict '{class_names[majority]}')")
    print(f"Trained KNN Model Accuracy : {accuracy:.4f}")

    if accuracy > baseline:
        print("\n[Conclusion] Model successfully learned useful patterns from the data (Beats Baseline).")
    else:
        print("\n[Conclusion] Model performs close to or below the baseline guessing.")

    # -----------------------------------------------------------------------
    # Save Predictions
    # -----------------------------------------------------------------------
    title("Saving Output Files")
    evaluate.save_predictions(y_test, y_pred, class_names, OUT_DIR / "predictions.csv")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"   - outputs/{f.name}")

if __name__ == "__main__":
    main()