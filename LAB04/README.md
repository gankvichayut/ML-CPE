# 📊 LAB 4: K-Nearest Neighbors (KNN) & Clustering (Heart Disease Dataset)

โครงการนี้เป็นส่วนหนึ่งของวิชา **Machine Learning (04-624-201)**  
ภาควิชาวิศวกรรมคอมพิวเตอร์ คณะวิศวกรรมศาสตร์ มหาวิทยาลัยเทคโนโลยีราชมงคลธัญบุรี

---

## 🎯 วัตถุประสงค์
1. เพื่อประยุกต์ใช้อัลกอริทึม K-Nearest Neighbors (KNN) ในการจำแนกประเภทข้อมูลทางการแพทย์ (Classification)
2. เพื่อเปรียบเทียบประสิทธิภาพของโมเดลเมื่อปรับเปลี่ยนจำนวนเพื่อนบ้าน (k values)
3. เพื่อจัดกลุ่มข้อมูลผู้ป่วยด้วย K-Means Clustering โดยใช้ Elbow Method และ Silhouette Score (Clustering)
4. เพื่อศึกษาความสำคัญของการทำ Feature Standardization ต่อโมเดลที่อาศัยระยะทาง (Distance-based Algorithms)

---

## 📁 โครงสร้างไฟล์ใน Repository

ML-04-KNN/  
│  
├── classification/  
│   ├── dataset_heart.csv          # ชุดข้อมูลโรคหัวใจสำหรับงาน Classification  
│   ├── main.py                    # สคริปต์หลักสำหรับเทรนและประเมินผล KNN  
│   ├── data_loader.py             # โหลดข้อมูล แยก Train/Val/Test และทำ Standardization  
│   ├── knn_tf.py                  # คลาสโมเดล KNN บน TensorFlow  
│   ├── evaluate.py                # ฟังก์ชันวัดผล กราฟ k-curve และ Confusion Matrix  
│   └── outputs/  
│       ├── 01_k_curve.png         # กราฟเปรียบเทียบค่า k กับ Validation Accuracy  
│       ├── 02_confusion_matrix.png # ภาพ Confusion Matrix บน Test Set  
│       └── predictions.csv        # ผลการทำนายเทียบกับค่าจริง  
│  
├── clustering/  
│   ├── dataset_heart.csv          # ชุดข้อมูลโรคหัวใจสำหรับงาน Clustering  
│   ├── main.py                    # สคริปต์หลักสำหรับรัน K-Means และจับคู่กลุ่มด้วย KNN  
│   ├── data_loader.py             # โหลดข้อมูลและทำ Standardization เฉพาะ Features  
│   ├── kmeans_tf.py               # คลาส K-Means Clustering บน TensorFlow  
│   ├── knn_tools.py               # คลาส KNN สำหรับกำหนด Cluster ให้ข้อมูลใหม่  
│   ├── visualize.py               # ฟังก์ชันพล็อต Elbow Curve และ Scatter Plot  
│   └── outputs/  
│       ├── 01_elbow.png           # กราฟ Elbow Method หาค่า k ที่เหมาะสม  
│       ├── 02_clusters.png        # กราฟกระจายตัวของ Cluster (Age vs Max Heart Rate)  
│       ├── cluster_summary.csv    # สรุปค่าเฉลี่ยสถิติของแต่ละ Cluster  
│       └── clustered_heart_data.csv # ข้อมูลพร้อมระบุ Cluster ของผู้ป่วยแต่ละคน  
│  
├── requirements.txt               # รายการไลบรารีที่จำเป็น  
└── README.md                      # เอกสารอธิบายโครงการ  

---

## 🛠️ เครื่องมือและไลบรารีที่ใช้ (Technologies Used)
* **Python** 3.x
* **TensorFlow** - การคำนวณ Tensor, Matrix Operations และ Euclidean Distance
* **Scikit-Learn** - การแบ่งชุดข้อมูล (Train/Val/Test), Feature Standardization และการประเมินผล
* **Pandas & NumPy** - การจัดการตารางข้อมูลและการคำนวณเชิงตัวเลข
* **Matplotlib** - การสร้างกราฟและการแสดงผลข้อมูล (Data Visualization)

---

## ⚙️ ขั้นตอนการดำเนินงาน (Methodology)

### 1. Classification (K-Nearest Neighbors)
* **Data Preparation & Scaling:** โหลดชุดข้อมูล 270 แถว และทำ Standardization (Z-score Scaling) ให้ Mean = 0, Std = 1
* **Data Splitting:** แบ่งข้อมูลออกเป็น Train 60%, Validation 20% และ Test 20% แบบ Stratified
* **Hyperparameter Tuning:** ทดสอบหาค่าเพื่อนบ้าน ($k$) ที่เหมาะสมบน Validation Set ($k \in [1, 3, 5, 7, 9, 11, 15, 21]$)
* **Evaluation:** ประเมินโมเดลบน Test Set ด้วยค่า Accuracy, Classification Report, Confusion Matrix และเปรียบเทียบผลกับ Scikit-Learn รวมถึง Zero-Rule Baseline

### 2. Clustering (K-Means & KNN Assignment)
* **Feature Extraction:** นำ Features ทางการแพทย์ทั้ง 13 ตัวมาทำการ Standardize โดยไม่ใช้ Target Label
* **Optimal Cluster Search:** วิเคราะห์จำนวนกลุ่มที่เหมาะสมผ่าน **Elbow Method** (ค่า Inertia) และ **Silhouette Score**
* **K-Means Execution:** รัน K-Means แบ่งข้อมูลออกเป็นกลุ่ม ($k = 2$) พร้อมวิเคราะห์คุณลักษณะเฉลี่ยของแต่ละกลุ่ม
* **KNN Cluster Assignment:** จำลองการนำข้อมูลผู้ป่วยใหม่เข้ามา แล้วใช้โมเดล KNN กำหนดกลุ่มให้อัตโนมัติ

---

## 📈 ผลลัพธ์และข้อสรุป (Results & Insights)
* **ประสิทธิภาพโมเดล KNN Classification:** โมเดลทำนายบน Test Set ได้ค่าความแม่นยำ (Accuracy) อยู่ที่ **79.63%** ซึ่งสูงกว่าการสุ่มเดาตามคลาสเสียงข้างมาก (Baseline Guessing)
* **การจัดกลุ่ม K-Means Clustering:** 
  * **Cluster 0 (เสี่ยงต่ำ):** อายุเฉลี่ย 52 ปี, อัตราการเต้นหัวใจสูงสุดเฉลี่ย 159.9 bpm, มีอาการเจ็บหน้าอกขณะออกกำลังกายเพียง 11.9% (ตรงกับกลุ่มไม่พบโรคจริง 81.5%)
  * **Cluster 1 (เสี่ยงสูง):** อายุเฉลี่ย 58.4 ปี, อัตราการเต้นหัวใจสูงสุดเฉลี่ย 132.8 bpm, มีอาการเจ็บหน้าอกขณะออกกำลังกายสูงถึง 67.6% (ตรงกับกลุ่มตรวจพบโรคจริง 87.3%)
* **ความสำคัญของ Feature Standardization:** เนื่องจากตัวแปรทางการแพทย์มีหน่วยและช่วงสเกลต่างกันมาก การ Standardize ช่วยป้องกันไม่ให้ตัวแปรที่มีค่ามากครอบงำผลการคำนวณระยะทาง Euclidean Distance

---

## 👤 จัดทำโดย
* **ชื่อ-นามสกุล:** วิชยุตม์ แก้ววิเศษ
* **รหัสนักศึกษา:** 116710400582-8
* **สาขา:** วิศวกรรมคอมพิวเตอร์
