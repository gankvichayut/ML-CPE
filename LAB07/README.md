# LAB 7: Convolutional Neural Network (CNN) for Image Recognition (Rock-Paper-Scissors Dataset)

โครงการนี้เป็นส่วนหนึ่งของวิชา **Machine Learning (04-624-201)**
ภาควิชาวิศวกรรมคอมพิวเตอร์ คณะวิศวกรรมศาสตร์ มหาวิทยาลัยเทคโนโลยีราชมงคลธัญบุรี

---

# Data

Kaggle Dataset: https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors/

---

## วัตถุประสงค์
1. เพื่อให้นักศึกษาเข้าใจสถาปัตยกรรมและหลักการทำงานของ Convolutional Neural Network (CNN) เช่น Convolutional Layer, Pooling Layer, และ Flatten/Dense Layer ในการจำแนกประเภทรูปภาพ (Image Classification)
2. เพื่อฝึกฝนกระบวนการเตรียมและประมวลผลข้อมูลภาพ (Computer Vision Preprocessing) เช่น การปรับขนาดภาพ (Resize เป็น 100x100), การแปลงปริภูมิสีเป็น RGB และการทำ Data Augmentation (Flip, Rotation) เพื่อเพิ่มความทนทานของโมเดล
3. เพื่อสร้างและเปรียบเทียบประสิทธิภาพของโครงสร้างโมเดล CNN หลากหลายรูปแบบ (CNN Configurations) เช่น จำนวน Convolutional Layer (1 ถึง 4 ชั้น) และจำนวนหน่วยประมวลผลใน Fully-Connected Layer (Dense Units)
4. เพื่อศึกษาและวิเคราะห์ผลกระทบของจำนวนรอบการฝึกสอน (Epochs) ต่อค่าความแม่นยำ (Validation Accuracy) และความเร็วในการลู่เข้าของแต่ละแบบจำลอง (Convergence Rate)
5. เพื่อประเมินประสิทธิภาพของแบบจำลองที่ดีที่สุดด้วยค่า Test Accuracy, Classification Report, Confusion Matrix และทดสอบการพยากรณ์ผลภาพตัวอย่างแบบสุ่มพร้อมระดับความเชื่อมั่น (Confidence Score)

---

## โครงสร้างไฟล์ใน Repository
```text
LAB07/
│
├── rps-cv-images/                 # โฟลเดอร์ชุดข้อมูลภาพเป่ายิ้งฉุบ (Raw Image Dataset)
│   ├── paper/                     # ภาพคลาสกระดาษ (Paper)
│   ├── rock/                      # ภาพคลาสค้อน (Rock)
│   ├── scissors/                  # ภาพคลาสกรรไกร (Scissors)
│   └── README_rpc-cv-images.txt   # คำอธิบายชุดข้อมูลภาพ
│
├── m-project/
│   ├── data_loader.py             # สคริปต์โหลดรูปภาพจากโฟลเดอร์คลาสและแปลงเป็นอาร์เรย์
│   ├── preprocessing.py           # ฟังก์ชัน Resize ภาพ (100x100) และแปลงสี BGR เป็น RGB
│   ├── split_data.py              # แบ่งข้อมูล Train (70%), Validation (10%), Test (20%) แบบ Stratified
│   ├── cnn_model.py               # สถาปัตยกรรม CNN ปรับแต่งได้ (Conv2D, BatchNorm, MaxPool) และฟังก์ชัน Train
│   ├── evaluate.py                # ฟังก์ชันคำนวณ Metrics, วาด Confusion Matrix และกราฟเปรียบเทียบโมเดล
│   ├── main.py                    # สคริปต์หลักรัน Pipeline การเทรนเปรียบเทียบ Configurations และบันทึกผล
│   ├── test_cnn.py                # สคริปต์สุ่มรูปภาพ 4 ภาพมาทดสอบพยากรณ์และแสดงผลกริด 2x2
│   │
│   └── outputs/                   # โฟลเดอร์เก็บผลลัพธ์และโมเดลที่บันทึก >25MB
│       ├── classes.json           # รายชื่อคลาส ['paper', 'rock', 'scissors']
│       ├── features.npy           # ข้อมูลรูปภาพทั้งหมดในรูป NumPy Array
│       ├── labels.npy             # ข้อมูล Labels ทั้งหมด
│       ├── X_train.npy / X_val.npy / X_test.npy # ข้อมูลแบ่งชุดรูปภาพ
│       ├── y_train.npy / y_val.npy / y_test.npy # ข้อมูลแบ่งชุด Label
│       ├── cnn_model.keras        # โมเดล CNN ที่ดีที่สุดที่บันทึกในรูปแบบ Keras
│       ├── confusion_matrix.png   # ภาพแผนภูมิความร้อน Confusion Matrix ของโมเดลที่ดีที่สุด
│       ├── models_comparison.png  # กราฟเปรียบเทียบ Training Performance (Validation Accuracy vs Epochs)
│       └── prediction_sample.png  # ผลการสุ่มทดสอบพยากรณ์ภาพ 4 ตัวอย่าง
│
└── README.md                      # เอกสารสรุปและอธิบายผลการทดลอง
