"""
data_loader.py (Clustering)
หน้าที่:
1. โหลดข้อมูล Heart Disease จากไฟล์ CSV
2. สกัดเอาเฉพาะ Features ทางการแพทย์ (13 ฟีเจอร์) โดยตัดคอลัมน์ Target ออก
3. ทำ Standard Scaling (Z-score) เพื่อให้ K-Means คำนวณระยะห่างได้อย่างเป็นธรรม
"""

from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ชี้ Path ไปยังไฟล์ dataset_heart.csv ภายในโฟลเดอร์ clustering
CSV_PATH = Path(__file__).resolve().parent / "dataset_heart.csv"

# กำหนดคอลัมน์ Features ทั้งหมด 13 ตัว
FEATURES = [
    "age",
    "sex",
    "chest pain type",
    "resting blood pressure",
    "serum cholestoral",
    "fasting blood sugar",
    "resting electrocardiographic results",
    "max heart rate",
    "exercise induced angina",
    "oldpeak",
    "ST segment",
    "major vessels",
    "thal",
]

def load_data():
    """
    คืนค่าเป็น Dictionary:
        X        : ข้อมูลหลัง Standardization (Mean=0, Std=1) ใช้สำหรับ K-Means / Distance
        X_raw    : ข้อมูลค่าจริงดั้งเดิม (ใช้วิเคราะห์ค่าเฉลี่ยทางกายภาพของแต่ละ Cluster)
        df       : DataFrame ข้อมูลดั้งเดิมทั้งหมด
        features : รายชื่อคอลัมน์ Features ที่นำมาใช้
    """
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()  # ตัด space ส่วนเกิน เช่น 'sex '
    df = df.dropna()

    X_raw = df[FEATURES].to_numpy(dtype="float32")
    # Standardize ให้ทุกมิติมีสเกลเท่ากัน ป้องกันค่าวัดหลักร้อยกลืนค่าทศนิยม
    X = StandardScaler().fit_transform(X_raw).astype("float32")

    return {"X": X, "X_raw": X_raw, "df": df, "features": FEATURES}

if __name__ == "__main__":
    data = load_data()
    print("ขนาดชุดข้อมูล (N, Features):", data["X"].shape)
    print("ค่าเฉลี่ยหลังทำ Scaling (ควรใกล้เคียง 0):", data["X"].mean(axis=0).round(3))