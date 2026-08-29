

# LAB 5: Support Vector Machine (SVM) (Breast Cancer Dataset)

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
│   └── brca.csv                         # ไฟล์ชุดข้อมูลมะเร็งเต้านม (Breast Cancer Wisconsin Diagnostic) สำหรับนำมาฝึกสอนและทดสอบโมเดล
│
├── mini-project/
│   ├── data_load.py                     # สคริปต์โหลดข้อมูลจาก CSV, จัดการคอลัมน์ที่ไม่จำเป็น และแปลง Target (Benign=0, Malignant=1)
│   ├── split_data.py                    # สคริปต์แบ่งข้อมูลออกเป็น Train Set (80%) และ Test Set (20%) แบบ Stratified
│   ├── preprocess.py                    # สคริปต์ปรับสเกลข้อมูลฟีเจอร์ด้วย StandardScaler (Z-score Normalization)
│   ├── svm_model.py                     # สคริปต์สร้างและฝึกสอนโมเดล Support Vector Classifier (SVC)
│   ├── evaluate.py                      # สคริปต์ประเมินผลความแม่นยำ (Accuracy, Report) และวาดแผนภาพ Confusion Matrix
│   ├── main.py                          # สคริปต์หลักที่รัน Pipeline ทั้งหมด เปรียบเทียบ 3 เคอร์เนล และบันทึก Best Model
│   ├── test_svm.py                      # สคริปต์สำหรับโหลด Best Model มาสุ่มตัวอย่างทำนายผลข้อมูลใหม่
│   │
│   └── outputs/                         # โฟลเดอร์เก็บผลลัพธ์และอ็อบเจกต์ที่ได้จากการรันโปรแกรม
│       ├── classes.json                 # ไฟล์บันทึกรายชื่อคลาสเป้าหมาย ['Benign', 'Malignant']
│       ├── confusion_matrix_linear.png  # ภาพผลลัพธ์ Confusion Matrix ของ Linear Kernel
│       ├── confusion_matrix_poly.png    # ภาพผลลัพธ์ Confusion Matrix ของ Polynomial Kernel
│       ├── confusion_matrix_rbf.png     # ภาพผลลัพธ์ Confusion Matrix ของ RBF Kernel
│       ├── scaler.pkl                   # อ็อบเจกต์ StandardScaler ที่ Fit แล้ว สำหรับนำไปแปลงข้อมูลชุดใหม่
│       ├── svm_model.pkl                # โมเดล SVM ที่ผ่านการเทรนและได้ค่าความแม่นยำสูงที่สุด (Best Model)
│       ├── X_train.npy / X_test.npy     # ข้อมูล Features ที่ผ่านการปรับสเกลแล้ว บันทึกในรูป NumPy Array
│       └── y_train.npy / y_test.npy     # ข้อมูล Labels ของชุดฝึกสอนและชุดทดสอบ
│
└── README.md                            # เอกสารสรุปรายละเอียดโครงการ วัตถุประสงค์ และผลการทดลอง
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
