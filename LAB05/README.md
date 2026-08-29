

# LAB 5: Support Vector Machine (SVM) on Breast Cancer Dataset

โครงการนี้เป็นส่วนหนึ่งของวิชา **Machine Learning (04-624-201)**  
ภาควิชาวิศวกรรมคอมพิวเตอร์ คณะวิศวกรรมศาสตร์ มหาวิทยาลัยเทคโนโลยีราชมงคลธัญบุรี

---

# Data

Kaggle Dataset: https://www.kaggle.com/datasets/utkarshx27/breast-cancer-wisconsin-diagnostic-dataset

---

## วัตถุประสงค์
1. เพื่อให้นักศึกษาเข้าใจหลักการทำงานของ Support Vector Machine (SVM) และแนวทางการประยุกต์ใช้ในงานจริงทั้งด้านการจำแนกและการพยากรณ์
2. เพื่อให้นักศึกษาสามารถพัฒนาผังการทำงาน ประมวลผล และแสดงผลลัพธ์ เพื่อพัฒนาระบบต้นแบบร่วมกับ SVM ในการจำแนกและการพยากรณ์ได้อย่างถูกต้อง
3. เพื่อประยุกต์ใช้ Support Vector Machine (SVM) ในการจำแนกประเภทข้อมูลทางการแพทย์จากชุดข้อมูล Breast Cancer Wisconsin Diagnostic Dataset (`brca.csv`)
4. เพื่อศึกษาและเปรียบเทียบประสิทธิภาพของแบบจำลอง SVM เมื่อใช้ฟังก์ชันเคอร์เนล (Kernel Functions) รูปแบบต่างๆ ได้แก่ Linear, Polynomial (Poly) และ Radial Basis Function (RBF)
5. เพื่อให้นักศึกษาสามารถเขียนโปรแกรมด้วยภาษา Python และประเมินประสิทธิภาพของแบบจำลองด้วยค่า Accuracy, Precision, Recall, F1-Score และ Confusion Matrix รวมถึงนำเสนอผลงานผ่าน GitHub

---

## โครงสร้างไฟล์ใน Repository
```text
LAB05/
│
├── dataset/
│   └── brca.csv                   # ชุดข้อมูล Breast Cancer Wisconsin (Diagnostic)
│
├── mini-project/
│   ├── main.py                    # สคริปต์หลักรัน Pipeline ฝึกสอนและเปรียบเทียบเคอร์เนล
│   ├── test_svm.py                # สคริปต์สุ่มตัวอย่างทดสอบการทำนายผลจาก Best Model
│   ├── data_load.py               # โหลดข้อมูล แปลง Label (Benign=0, Malignant=1) และสกัด Features
│   ├── preprocess.py              # ปรับสเกลข้อมูล Features ด้วย StandardScaler
│   ├── split_data.py              # แบ่งข้อมูล Train 80% และ Test 20% แบบ Stratified
│   ├── svm_model.py               # สร้างและฝึกสอนแบบจำลอง Support Vector Classifier (SVC)
│   ├── evaluate.py                # ประเมินประสิทธิภาพและสร้างภาพ Confusion Matrix
│   └── outputs/
│       ├── . . .
│
└── README.md                      # เอกสารอธิบายและสรุปผลการทดลอง
```

---

## ผลลัพธ์และข้อสรุป (Results & Insights)
* **ความสำคัญของ Feature Standardization:** เนื่องจากแบบจำลอง SVM ทำงานโดยการคำนวณระยะห่างทางเรขาคณิต (Margin) ระหว่าง Support Vectors และ Hyperplane การปรับสเกลข้อมูลทางการแพทย์ให้เป็นมาตรฐานเดียวกันจึงเป็นขั้นตอนสำคัญที่ช่วยเพิ่มความถูกต้องและลดความเอนเอียงของโมเดล
* **ประสิทธิภาพของ SVM Kernels:**
  * โมเดล **Linear Kernel** และ **RBF Kernel** สามารถสร้างขอบเขตการตัดสินใจ (Decision Boundary) ในการแยกผู้ป่วยกลุ่มเนื้อร้าย (Malignant) และกลุ่มเนื้อดี (Benign) ได้อย่างมีประสิทธิภาพสูง
  * การเลือกใช้ฟังก์ชันเคอร์เนลที่เหมาะสมกับโครงสร้างการกระจายตัวของฟีเจอร์ทางการแพทย์ ส่งผลต่อค่าความแม่นยำ (Accuracy) และอัตราการตรวจจับเนื้อร้าย (Recall) ของระบบ
* **การนำไปประยุกต์ใช้งาน:** แบบจำลอง SVM ที่ผ่านการฝึกสอนสามารถนำมาบันทึกและโหลดกลับมาใช้งานเป็นระบบต้นแบบในการช่วยวินิจฉัยและจำแนกข้อมูลผู้ป่วยใหม่ได้อย่างถูกต้องและรวดเร็ว

---

## จัดทำโดย
* **ชื่อ-นามสกุล:** วิชยุตม์ แก้ววิเศษ
* **รหัสนักศึกษา:** 116710400582-8
* **ภาควิชา:** วิศวกรรมคอมพิวเตอร์
