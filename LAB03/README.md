# LAB 3: Regression & Classification (Face Analysis Dataset)

งานนี้เป็นส่วนหนึ่งของวิชา **Machine Learning (04-624-201)**  
ภาควิชาวิศวกรรมคอมพิวเตอร์ คณะวิศวกรรมศาสตร์ มหาวิทยาลัยเทคโนโลยีราชมงคลธัญบุรี

---

# Data

Kaggle Dataset: https://www.kaggle.com/datasets/nipunarora8/age-gender-and-ethnicity-face-data-csv

---

## วัตถุประสงค์
1. เพื่อให้นักศึกษาเข้าใจหลักการของ Regression และ Classification ซึ่งเป็นเทคนิคพื้นฐานของ Supervised Learning และสามารถอธิบายความแตกต่างระหว่างการทำนายค่าต่อเนื่อง (Continuous Value) และการจำแนกประเภทข้อมูลได้
2. เพื่อให้นักศึกษาสามารถเตรียมข้อมูลสำหรับการสร้างแบบจำลอง โดยเลือกใช้คุณลักษณะ (Features) ที่เหมาะสม รวมถึงประยุกต์ใช้เทคนิค Principal Component Analysis เพื่อลดจำนวนคุณลักษณะและเพิ่มประสิทธิภาพของการเรียนรู้
3. เพื่อให้นักศึกษาสามารถพัฒนาแบบจำลอง Linear Regression สำหรับการทำนายอายุจากภาพใบหน้า และแบบจำลอง Classification สำหรับการจำแนกเพศจากภาพใบหน้า พร้อมเปรียบเทียบผลลัพธ์ของแต่ละวิธี
4. เพื่อให้นักศึกษาสามารถเขียนโปรแกรมด้วยภาษา Python และใช้ไลบรารีด้าน Machine Learning ในการสร้าง ฝึกสอน (Training) ทดสอบ (Testing) และประเมินประสิทธิภาพของแบบจำลอง
5. เพื่อให้นักศึกษาสามารถวิเคราะห์และอธิบายผลลัพธ์ของแบบจำลองด้วยตัวชี้วัดที่เหมาะสม เช่น MSE, MAE, R² Score, Accuracy, Precision, Recall, F1-score, ROC Curve และ AUC รวมถึงนำเสนอผลงานผ่าน GitHub

---

## โครงสร้างไฟล์ใน Repository
```text
ML-03-Regression-Classification/
│
├── age_gender.csv          # ชุดข้อมูลภาพใบหน้า UTKFace (Features & Labels) (ดาวโหลดจากลิงก์ Kaggle)
├── LAB3_code.ipynb         # ไฟล์ Jupyter Notebook หลักที่รวมโค้ดและผลการทดลองทั้งหมด  
└── README.md               # เอกสารอธิบายและสรุปผลการทดลอง
```

---

## ผลลัพธ์และข้อสรุป (Results & Insights)
* **การลดมิติข้อมูลด้วย PCA:** การลดขนาดมิติจาก 2,304 พิกเซลเหลือ 50 Components ช่วยรักษาความแปรปรวนหลักของโครงสร้างใบหน้าได้อย่างมีประสิทธิภาพและลดสัญญาณรบกวน (Noise)
* **ประสิทธิภาพงาน Regression (ทำนายอายุ):** แบบจำลอง Multiple Linear Regression สามารถทำนายอายุได้แม่นยำกว่า Simple Linear Regression เนื่องจากได้รับข้อมูลองค์ประกอบใบหน้าจากหลายมิติพร้อมกัน
* **ประสิทธิภาพงาน Classification (จำแนกเพศ):** แบบจำลอง Logistic Regression สามารถสร้างระนาบ Decision Boundary เพื่อจำแนกเพศได้ชัดเจน พร้อมให้ค่า ROC-AUC และ F1-Score ในระดับที่น่าพึงพอใจ
* **ความแตกต่างระหว่าง Regression และ Classification:** 
  * **Regression:** มุ่งเน้นการลดระยะห่างของค่าคลาดเคลื่อนเชิงปริมาณ ($y \in \mathbb{R}$) ประเมินผลด้วย Residuals (MSE, MAE, $R^2$)
  * **Classification:** มุ่งเน้นการหากฎเกณฑ์ในการแบ่งแยกกลุ่มความน่าจะเป็น ($P(y=1)$) ประเมินผลด้วย Confusion Matrix, Precision, Recall, F1 และ ROC-AUC

---

## จัดทำโดย
* **ชื่อ-นามสกุล:** วิชยุตม์ แก้ววิเศษ
* **รหัสนักศึกษา:** 116710400582-8
* **ภาควิชา:** วิศวกรรมคอมพิวเตอร์
