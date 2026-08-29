# LAB 6: Neural Network (MLP) for Image Recognition (Rock-Paper-Scissors Dataset)

โครงการนี้เป็นส่วนหนึ่งของวิชา **Machine Learning (04-624-201)**
ภาควิชาวิศวกรรมคอมพิวเตอร์ คณะวิศวกรรมศาสตร์ มหาวิทยาลัยเทคโนโลยีราชมงคลธัญบุรี

---

# Data

Kaggle Dataset: https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors/

---

## วัตถุประสงค์
1. เพื่อให้นักศึกษาเข้าใจสถาปัตยกรรมและหลักการทำงานของ Artificial Neural Network (Multilayer Perceptron: MLP) ในการจำแนกประเภทรูปภาพ (Image Classification)
2. เพื่อฝึกฝนกระบวนการเตรียมและประมวลผลข้อมูลภาพ (Computer Vision Preprocessing) เช่น การปรับขนาดภาพ (Resize), การแปลงปริภูมิสี (BGR to RGB) และการจัดเก็บโครงสร้าง Tensor สำหรับโมเดล Deep Learning
3. เพื่อสร้างและฝึกสอนโครงข่ายประสาทเทียมด้วย TensorFlow / Keras โดยประยุกต์ใช้เทคนิค Batch Normalization, Dropout เพื่อป้องกัน Overfitting และ Early Stopping กับ ReduceLROnPlateau ในการปรับอัตราการเรียนรู้
4. เพื่อศึกษาและวิเคราะห์เส้นโค้งการเรียนรู้ (Learning Curves: Loss & Accuracy) ทั้งบน Training Set และ Validation Set เพื่อประเมินภาวะ Overfitting/Underfitting
5. เพื่อประเมินประสิทธิภาพของแบบจำลองด้วยค่า Accuracy, Classification Report (Precision, Recall, F1-Score), Confusion Matrix และทดสอบการพยากรณ์ผลภาพตัวอย่างแบบสุ่มพร้อมระดับความเชื่อมั่น (Confidence Score)

---

## โครงสร้างไฟล์ใน Repository
```text
LAB06/
│
├── rps-cv-images/                                 # โฟลเดอร์ชุดข้อมูลภาพเป่ายิ้งฉุบ (Raw Image Dataset) (ดาวโหลดจากลิงก์ Kaggle)
│   ├── paper/                                     # ภาพคลาสกระดาษ (Paper)
│   ├── rock/                                      # ภาพคลาสค้อน (Rock)
│   ├── scissors/                                  # ภาพคลาสกรรไกร (Scissors)
│   └── README_rpc-cv-images.txt                   # คำอธิบายชุดข้อมูลภาพ
│
├── mini-project/
│   ├── data_loader.py                             # สคริปต์โหลดรูปภาพจากโฟลเดอร์คลาสและแปลงเป็นอาร์เรย์
│   ├── preprocessing.py                           # ฟังก์ชัน Resize ภาพ (64x64) และจัดการรูปแบบสี RGB
│   ├── split_data.py                              # แบ่งข้อมูล Train (70%), Validation (10%), Test (20%) แบบ Stratified
│   ├── nn_model.py                                # สถาปัตยกรรมโมเดล Sequential MLP, การ Rescaling และกระบวนการ Fit
│   ├── evaluate.py                                # ฟังก์ชันคำนวณ Metrics, วาด Confusion Matrix และกราฟ History Curves
│   ├── main.py                                    # สคริปต์หลักรัน Pipeline การเทรนและบันทึกโมเดลทั้งหมด
│   ├── test_nn.py                                 # สคริปต์สุ่มรูปภาพ 4 ภาพมาทดสอบพยากรณ์และแสดงผลกริด 2x2
│   │
│   └── outputs/                                   # โฟลเดอร์เก็บผลลัพธ์และโมเดลที่บันทึก
│       ├── classes.json                           # รายชื่อคลาส ['paper', 'rock', 'scissors']
│       ├── history.json                           # ประวัติค่า Loss และ Accuracy ในแต่ละ Epoch
│       ├── features.npy                           # ข้อมูล Features ทั้งหมดในรูป NumPy Array
│       ├── labels.npy                             # ข้อมูล Labels ทั้งหมด
│       ├── X_train.npy / X_val.npy / X_test.npy   # ข้อมูลแบ่งชุดรูปภาพ
│       ├── y_train.npy / y_val.npy / y_test.npy   # ข้อมูลแบ่งชุด Label
│       ├── nn_model.keras                         # โมเดล Neural Network ที่บันทึกในรูปแบบ Keras
│       ├── confusion_matrix.png                   # ภาพแผนภูมิความร้อน Confusion Matrix
│       ├── training_history.png                   # กราฟเปรียบเทียบ Training & Validation (Loss / Accuracy)
│       └── prediction_sample.png                  # ผลการสุ่มทดสอบพยากรณ์ภาพ 4 ตัวอย่าง
│
└── README.md                                      # เอกสารสรุปและอธิบายผลการทดลอง
```

---

## ผลลัพธ์และข้อสรุป (Results & Insights)
* **ประสิทธิภาพของการ Regularization:** การเพิ่มเลเยอร์ BatchNormalization และ Dropout ช่วยควบคุมไม่ให้โครงข่ายประสาทขนาดใหญ่เกิดปัญหา Overfitting บนชุดข้อมูลภาพที่มีมิติสูง ($12,288$ ฟีเจอร์)
* **บทบาทของ Validation Set & Early Stopping:** การติดตามผลด้วย Validation Set ร่วมกับการใช้ Early Stopping ช่วยให้โมเดลหยุดการทำงานที่จุดที่ดีที่สุด พร้อมกู้คืนค่าน้ำหนัก (Weights) ที่ให้ผล Generalization บนข้อมูลใหม่ได้อย่างแม่นยำ
* **ความแม่นยำในการจำแนกท่ามือ (RPS):** แบบจำลอง Fully-Connected Neural Network สามารถแยกแยะความแตกต่างของลักษณะท่ามือเป่ายิ้งฉุบทั้ง 3 คลาสได้อย่างมีนัยสำคัญ และสามารถให้ค่าความเชื่อมั่นที่แม่นยำในการทดสอบตัวอย่างใหม่

---

## จัดทำโดย
* **ชื่อ-นามสกุล:** วิชยุตม์ แก้ววิเศษ
* **รหัสนักศึกษา:** 116710400582-8
* **ภาควิชา:** วิศวกรรมคอมพิวเตอร์
