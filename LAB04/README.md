# LAB 4: K-Nearest Neighbors (KNN) & Clustering (Heart Disease Dataset)

งานนี้เป็นส่วนหนึ่งของวิชา **Machine Learning (04-624-201)**  
ภาควิชาวิศวกรรมคอมพิวเตอร์ คณะวิศวกรรมศาสตร์ มหาวิทยาลัยเทคโนโลยีราชมงคลธัญบุรี

---

# Data

Kaggle Dataset: https://www.kaggle.com/datasets/utkarshx27/heart-disease-diagnosis-dataset

---

## วัตถุประสงค์
1. เพื่อประยุกต์ใช้อัลกอริทึม K-Nearest Neighbors (KNN) ในการจำแนกประเภทข้อมูลทางการแพทย์ (Classification)
2. เพื่อเปรียบเทียบประสิทธิภาพของโมเดลเมื่อปรับเปลี่ยนจำนวนเพื่อนบ้าน (k values)
3. เพื่อจัดกลุ่มข้อมูลผู้ป่วยด้วย K-Means Clustering โดยใช้ Elbow Method และ Silhouette Score (Clustering)
4. เพื่อศึกษาความสำคัญของการทำ Feature Standardization ต่อโมเดลที่อาศัยระยะทาง (Distance-based Algorithms)

---

## โครงสร้างไฟล์ใน Repository
```text
LAB04/
│
├── classification/
│   ├── dataset_heart.csv             # ชุดข้อมูลโรคหัวใจสำหรับงาน Classification
│   ├── main.py                       # สคริปต์หลักสำหรับเทรนและประเมินผล KNN
│   ├── data_loader.py                # โหลดข้อมูล แยก Train/Val/Test และทำ Standardization
│   ├── knn_tf.py                     # คลาสโมเดล KNN บน TensorFlow
│   ├── evaluate.py                   # ฟังก์ชันวัดผล กราฟ k-curve และ Confusion Matrix
│   └── outputs/
│       ├── 01_k_curve.png            # กราฟเปรียบเทียบค่า k กับ Validation Accuracy
│       ├── 02_confusion_matrix.png   # ภาพ Confusion Matrix บน Test Set
│       └── predictions.csv           # ผลการทำนายเทียบกับค่าจริง
│
├── clustering/
│   ├── dataset_heart.csv             # ชุดข้อมูลโรคหัวใจสำหรับงาน Clustering
│   ├── main.py                       # สคริปต์หลักสำหรับรัน K-Means และจับคู่กลุ่มด้วย KNN
│   ├── data_loader.py                # โหลดข้อมูลและทำ Standardization เฉพาะ Features
│   ├── kmeans_tf.py                  # คลาส K-Means Clustering บน TensorFlow
│   ├── knn_tools.py                  # คลาส KNN สำหรับกำหนด Cluster ให้ข้อมูลใหม่
│   ├── visualize.py                  # ฟังก์ชันพล็อต Elbow Curve และ Scatter Plot
│   └── outputs/
│       ├── 01_elbow.png              # กราฟ Elbow Method หาค่า k ที่เหมาะสม
│       ├── 02_clusters.png           # กราฟกระจายตัวของ Cluster (Age vs Max Heart Rate)
│       ├── cluster_summary.csv       # สรุปค่าเฉลี่ยสถิติของแต่ละ Cluster
│       └── clustered_heart_data.csv  # ข้อมูลพร้อมระบุ Cluster ของผู้ป่วยแต่ละคน
│
└── README.md                         # เอกสารอธิบายโครงการ
```

---

## ผลลัพธ์และข้อสรุป (Results & Insights)
* **ประสิทธิภาพโมเดล KNN Classification:** โมเดลทำนายบน Test Set ได้ค่าความแม่นยำ (Accuracy) อยู่ที่ **79.63%** ซึ่งสูงกว่าการสุ่มเดาตามคลาสเสียงข้างมาก (Baseline Guessing)
* **การจัดกลุ่ม K-Means Clustering:** 
  * **Cluster 0 (เสี่ยงต่ำ):** อายุเฉลี่ย 52 ปี, อัตราการเต้นหัวใจสูงสุดเฉลี่ย 159.9 bpm, มีอาการเจ็บหน้าอกขณะออกกำลังกายเพียง 11.9% (ตรงกับกลุ่มไม่พบโรคจริง 81.5%)
  * **Cluster 1 (เสี่ยงสูง):** อายุเฉลี่ย 58.4 ปี, อัตราการเต้นหัวใจสูงสุดเฉลี่ย 132.8 bpm, มีอาการเจ็บหน้าอกขณะออกกำลังกายสูงถึง 67.6% (ตรงกับกลุ่มตรวจพบโรคจริง 87.3%)
* **ความสำคัญของ Feature Standardization:** เนื่องจากตัวแปรทางการแพทย์มีหน่วยและช่วงสเกลต่างกันมาก การ Standardize ช่วยป้องกันไม่ให้ตัวแปรที่มีค่ามากครอบงำผลการคำนวณระยะทาง Euclidean Distance

---

## จัดทำโดย
* **ชื่อ-นามสกุล:** วิชยุตม์ แก้ววิเศษ
* **รหัสนักศึกษา:** 116710400582-8
* **ภาควิชา:** วิศวกรรมคอมพิวเตอร์
