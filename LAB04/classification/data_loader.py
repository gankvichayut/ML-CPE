"""
data_loader.py
หน้าที่:
1. โหลดข้อมูล Heart Disease จากไฟล์ CSV
2. จัดการ Target Label (1: No Disease -> 0, 2: Disease -> 1)
3. แบ่งข้อมูล (Train 60% / Validation 20% / Test 20%)
4. ทำ Standardization ปรับสเกลข้อมูลให้พร้อมสำหรับโมเดล KNN
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# กำหนด Path ไปยังไฟล์ dataset_heart.csv (ชี้ไปที่โฟลเดอร์เดียวกัน)
CSV_PATH = Path(__file__).resolve().parent / "dataset_heart.csv"

# คอลัมน์เฉลย (Target)
TARGET = "heart disease"

def load_data(test_size=0.2, seed=42):
    # Step 1: อ่านไฟล์ CSV และตัดช่องว่างส่วนเกินที่หัวตาราง
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()  # ป้องกันปัญหา space เช่น 'sex '
    df = df.dropna()

    # Step 2: แยก Feature (X) และ Label (y)
    X = df.drop(columns=[TARGET]).copy()
    feature_names = list(X.columns)

    # แปลง Target: 1 -> 0 (No Disease), 2 -> 1 (Disease)
    class_names = ["No_Disease", "Disease"]
    y = df[TARGET].map({1: 0, 2: 1})

    X = X.to_numpy(dtype="float32")
    y = y.to_numpy(dtype="int32")

    # Step 3: แบ่งข้อมูล Train 60% / Val 20% / Test 20% ด้วย Stratified Split
    # รอบแรก: แยก Test ออกมา 20% (คงเหลือ 80% สำหรับ Train + Val)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # รอบสอง: แยก Train (60% ของทั้งหมด) และ Val (20% ของทั้งหมด หรือ 25% ของ X_temp)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=seed, stratify=y_temp
    )

    # Step 4: Standardization (Z-score Scaling) เพื่อให้ทุกฟีเจอร์มี Mean=0, Std=1
    # สำคัญมากสำหรับ KNN เพราะใช้ระยะทาง Euclidean Distance
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype("float32")
    X_val = scaler.transform(X_val).astype("float32")
    X_test = scaler.transform(X_test).astype("float32")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "class_names": class_names,
        "feature_names": feature_names,
        "n_rows": len(df),
    }

if __name__ == "__main__":
    data = load_data()
    print("train shape:", data["X_train"].shape)
    print("val shape  :", data["X_val"].shape)
    print("test shape :", data["X_test"].shape)
    print("classes    :", data["class_names"])