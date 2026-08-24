import json
import joblib
import numpy as np

OUTPUT_DIR = "outputs"

def test_svm(n_samples=5):
    model = joblib.load(f"{OUTPUT_DIR}/svm_model.pkl")
    X_test = np.load(f"{OUTPUT_DIR}/X_test.npy")
    y_test = np.load(f"{OUTPUT_DIR}/y_test.npy")
    with open(f"{OUTPUT_DIR}/classes.json") as f:
        classes = json.load(f)

    # สุ่มข้อมูลมาทดสอบ
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    X_sample = X_test[indices]
    y_sample = y_test[indices]

    predictions = model.predict(X_sample)

    print("\n--- Test Predictions Sample ---")
    for i in range(n_samples):
        pred_label = classes[predictions[i]]
        true_label = classes[y_sample[i]]
        status = "OK" if predictions[i] == y_sample[i] else "WRONG"
        print(f"Sample [{i+1}] -> Predicted: {pred_label:<10} | True: {true_label:<10} [{status}]")

if __name__ == "__main__":
    test_svm()