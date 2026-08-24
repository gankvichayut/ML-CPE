import json
import os
import joblib
import numpy as np

from data_load import load_data
from split_data import split_dataset
from preprocess import preprocess_features
from svm_model import train_svm, predict_svm
from evaluate import evaluate_model

DATA_PATH = r"C:\Users\vichay\Desktop\ML-CPE\LAB05\dataset\brca.csv"  # ชี้มาที่ไฟล์ CSV
OUTPUT_DIR = "outputs"
TEST_SIZE = 0.2

def main():
    print("=" * 60)
    print("SVM Classification: Breast Cancer Wisconsin Dataset")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load Dataset
    print("\n[Step 1] Loading dataset...")
    X, y, classes = load_data(DATA_PATH)

    with open(f"{OUTPUT_DIR}/classes.json", "w") as f:
        json.dump([str(c) for c in classes], f)

    # Step 2: Split Dataset
    print("\n[Step 2] Splitting dataset...")
    X_train, X_test, y_train, y_test = split_dataset(X, y, TEST_SIZE)

    # Step 3: Preprocessing (Standard Scaling)
    print("\n[Step 3] Preprocessing (Scaling)...")
    X_train_scaled, X_test_scaled, scaler = preprocess_features(X_train, X_test)

    # Save processed data & scaler
    np.save(f"{OUTPUT_DIR}/X_train.npy", X_train_scaled)
    np.save(f"{OUTPUT_DIR}/X_test.npy", X_test_scaled)
    np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)
    np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)
    joblib.dump(scaler, f"{OUTPUT_DIR}/scaler.pkl")

    # Step 4: Train & Compare 3 Kernels (Linear, Poly, RBF)
    print("\n[Step 4 & 5] Training & Evaluating 3 SVM Kernels...")
    kernels = ['linear', 'poly', 'rbf']
    best_acc = 0
    best_model = None
    best_kernel = ""

    for k in kernels:
        print(f"\n---> Kernel: {k.upper()} <---")
        model = train_svm(X_train_scaled, y_train, kernel=k)
        predictions = predict_svm(model, X_test_scaled)
        
        acc = evaluate_model(
            y_test, 
            predictions, 
            classes, 
            save_path=f"{OUTPUT_DIR}/confusion_matrix_{k}.png"
        )

        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_kernel = k

    # Save Best Model
    joblib.dump(best_model, f"{OUTPUT_DIR}/svm_model.pkl")
    print(f"\n==========================================")
    print(f"Best Kernel: {best_kernel.upper()} with Accuracy: {best_acc * 100:.2f}%")
    print(f"Saved best model to {OUTPUT_DIR}/svm_model.pkl")

if __name__ == "__main__":
    main()