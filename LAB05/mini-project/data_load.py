import pandas as pd
import numpy as np

def load_data(data_path="brca.csv"):
    df = pd.read_csv(data_path)
    
    # ดึง target column ออกมา (เช่น diagnosis หรือคอลัมน์สุดท้าย)
    if 'diagnosis' in df.columns:
        y_raw = df['diagnosis'].values
        # แปลง B (Benign)=0 และ M (Malignant)=1
        y = np.where(y_raw == 'M', 1, 0)
        X = df.drop(columns=['diagnosis', 'id'], errors='ignore').values
        classes = ['Benign', 'Malignant']
    else:
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        classes = sorted(list(np.unique(y)))

    print("Data loaded shape:", X.shape)
    print("Detected classes:", classes)

    return X, y, classes