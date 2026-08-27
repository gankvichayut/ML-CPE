"""
main.py (Clustering Pipeline)
ขั้นตอนการทำงาน:
STEP 1: โหลดและเตรียมข้อมูล (Scaled Features + Raw Features)
STEP 2: หาจำนวนกลุ่มที่เหมาะสม (Elbow Method & Silhouette Score)
STEP 3: รันโมเดล K-Means ด้วยค่า k ที่เลือก
STEP 4: วิเคราะห์คุณลักษณะทางสถิติ (Profile) ของแต่ละ Cluster
STEP 5: นำ KNN มาจำแนกผู้ป่วยกลุ่มใหม่เข้า Cluster ที่กำหนดไว้
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

import data_loader
import visualize
from kmeans_tf import TFKMeans
from knn_tools import KNNClusterAssigner

OUT_DIR = Path(__file__).resolve().parent / "outputs"

# จำนวนกลุ่มที่ต้องการแบ่ง (สำหรับโรคหัวใจ เลือก k = 2 หรือดูจาก Elbow Graph)
N_CLUSTERS = 2
KNN_K = 5

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
    X = data["X"]          # ข้อมูลหลัง Scale สำหรับคำนวณ
    X_raw = data["X_raw"]  # ข้อมูลจริง สำหรับอธิบายผล
    df = data["df"]
    features = data["features"]

    print(f"Data size: {X.shape[0]} rows x {X.shape[1]} features")
    print("Features used for clustering:")
    for f in features:
        print(f"   - {f}")

    # -----------------------------------------------------------------------
    # STEP 2 : Elbow Method & Silhouette Analysis
    # -----------------------------------------------------------------------
    title("STEP 2 : Determining Optimal Clusters (Elbow Method & Silhouette)")
    k_values = [2, 3, 4, 5, 6, 7, 8]
    inertias = []

    for k in k_values:
        km = TFKMeans(n_clusters=k).fit(X)
        sil = silhouette_score(X, km.labels_)
        inertias.append(km.inertia_)
        print(f"   k = {k}  -->  Inertia = {km.inertia_:8.1f} | Silhouette Score = {sil:.3f}")

    visualize.plot_elbow(k_values, inertias, OUT_DIR / "01_elbow.png")
    print(f"\n[Saved] Elbow curve saved to outputs/01_elbow.png")
    print(f"Selected k = {N_CLUSTERS} for clustering")

    # -----------------------------------------------------------------------
    # STEP 3 : Run K-Means
    # -----------------------------------------------------------------------
    title(f"STEP 3 : Execute K-Means (k = {N_CLUSTERS})")
    km = TFKMeans(n_clusters=N_CLUSTERS)
    labels = km.fit_predict(X)

    sil = silhouette_score(X, labels)
    print(f"Iterations until convergence : {km.n_iter_}")
    print(f"Final Inertia                : {km.inertia_:.1f}")
    print(f"Silhouette Score             : {sil:.3f}")
    print(f"Members per cluster          : {np.bincount(labels).tolist()}")

    # พลอตกระจายตัวของข้อมูลจริง: ตัวอย่างเลือก 'age' (คอลัมน์ 0) กับ 'max heart rate' (คอลัมน์ 7)
    visualize.plot_clusters(
        X_raw[:, [0, 7]], 
        labels, 
        OUT_DIR / "02_clusters.png",
        x_name="Age (years)", 
        y_name="Max Heart Rate"
    )

    # -----------------------------------------------------------------------
    # STEP 4 : Cluster Profiling & Characteristics
    # -----------------------------------------------------------------------
    title("STEP 4 : Cluster Characteristics & Statistical Summary")
    profile = pd.DataFrame(X_raw.astype("float64"), columns=features)
    profile["cluster"] = labels

    summary = profile.groupby("cluster").mean().round(2)
    summary["member_count"] = np.bincount(labels)

    print(summary.to_string())
    summary.to_csv(OUT_DIR / "cluster_summary.csv", encoding="utf-8-sig")

    # -----------------------------------------------------------------------
    # STEP 5 : Classify New Patients using KNN
    # -----------------------------------------------------------------------
    title(f"STEP 5 : Assign New Data to Clusters using KNN (k = {KNN_K})")
    # จำลองสถานการณ์: ใช้ 200 คนแรกเป็นกลุ่มที่กำหนด Cluster แล้ว และ 70 คนหลังเป็นคนไข้ใหม่
    n_known = 200
    X_known, labels_known = X[:n_known], labels[:n_known]
    X_new, labels_new = X[n_known:], labels[n_known:]

    assigner = KNNClusterAssigner(k=KNN_K)
    assigner.fit(X_known, labels_known)
    knn_pred = assigner.predict(X_new)

    accuracy = float(np.mean(knn_pred == labels_new))
    print(f"Number of new patients evaluated : {len(X_new)}")
    print(f"KNN Assignment Consistency Rate  : {accuracy * 100:.2f}%")

    # -----------------------------------------------------------------------
    # Save Final Clustered Data
    # -----------------------------------------------------------------------
    title("Saving Output Results")
    result = df.copy()
    result["cluster"] = labels
    result.to_csv(OUT_DIR / "clustered_heart_data.csv", index=False, encoding="utf-8-sig")

    for f in sorted(OUT_DIR.iterdir()):
        print(f"   - outputs/{f.name}")

if __name__ == "__main__":
    main()