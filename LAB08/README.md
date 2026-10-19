# LAB 8: Deep Convolutional Neural Network (DCNN) based on VGG Architecture for Image Recognition (Rock-Paper-Scissors Dataset)

โครงการนี้เป็นส่วนหนึ่งของวิชา **Machine Learning (04-624-201)**
ภาควิชาวิศวกรรมคอมพิวเตอร์ คณะวิศวกรรมศาสตร์ มหาวิทยาลัยเทคโนโลยีราชมงคลธัญบุรี

---

# Data

Kaggle Dataset: https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors/

---

## วัตถุประสงค์
1. เพื่อให้นักศึกษาเข้าใจสถาปัตยกรรมและหลักการทำงานของ Deep Convolutional Neural Network (DCNN) โดยเฉพาะสถาปัตยกรรมแบบ **VGG Network** ที่ใช้ Convolutional Blocks แบบซ้อนหลายชั้น (Small 3x3 Conv Filters) ร่วมกับ Batch Normalization และ Max Pooling ในการจำแนกประเภทรูปภาพ (Image Classification)
2. เพื่อฝึกฝนกระบวนการเตรียมและประมวลผลข้อมูลภาพ (Computer Vision Preprocessing) เช่น การปรับขนาดภาพ (Resize เป็น 50x50), การแปลงปริภูมิสีเป็น RGB และการทำ Data Augmentation ภายในโมเดล (Random Flip, Random Rotation, Random Zoom) เพื่อเพิ่มความทนทานและลดปัญหา Overfitting
3. เพื่อสร้างและฝึกสอนโมเดล Deep CNN ด้วยเทคนิคสมัยใหม่ เช่น การใช้ Global Average Pooling แทนการ Flatten แบบดั้งเดิม, การใส่ Dropout เพื่อป้องกัน Overfitting และการใช้ Callbacks (EarlyStopping, ReduceLROnPlateau) เพื่อควบคุมกระบวนการเทรนอย่างมีประสิทธิภาพ
4. เพื่อศึกษาและวิเคราะห์พฤติกรรมการเรียนรู้ของโครงข่ายประสาทเทียมระดับลึกผ่านจำนวนรอบการฝึกสอน (50 Epochs) โดยติดตามค่า Training/Validation Accuracy และ Loss เพื่อศึกษาอัตราการลู่เข้า (Convergence) และเสถียรภาพของโมเดล
5. เพื่อประเมินประสิทธิภาพของแบบจำลองด้วยค่า Test Accuracy, Classification Report, แผนภูมิความร้อน Confusion Matrix และทดสอบการพยากรณ์ภาพตัวอย่างแบบสุ่มพร้อมระดับความเชื่อมั่น (Confidence Score)

---

## โครงสร้างไฟล์ใน Repository
```text
ML-08-DCNN/
│
├── rps-cv-images/                 # โฟลเดอร์ชุดข้อมูลภาพเป่ายิ้งฉุบ (Raw Image Dataset)
│   ├── paper/                     # ภาพคลาสกระดาษ (Paper)
│   ├── rock/                      # ภาพคลาสค้อน (Rock)
│   ├── scissors/                  # ภาพคลาสกรรไกร (Scissors)
│   └── README_rpc-cv-images.txt   # คำอธิบายชุดข้อมูลภาพ
│
├── mini-proj/
│   ├── data_loader.py             # สคริปต์ตรวจจับคลาสอัตโนมัติและโหลดภาพจากโฟลเดอร์เป็นอาร์เรย์
│   ├── preprocessing.py           # ฟังก์ชัน Resize ภาพ (50x50), แปลงสี BGR เป็น RGB และจัดเตรียม Features
│   ├── split_data.py              # แบ่งข้อมูล Train (64%), Validation (16%), Test (20%) แบบ Stratified
│   ├── vgg_model.py               # โครงสร้าง VGG DCNN (VGG16 / VGG_SMALL), Data Augmentation และฟังก์ชัน Train
│   ├── evaluate.py                # ฟังก์ชันคำนวณ Metrics, วาด Confusion Matrix และกราฟ Training Performance & Loss
│   ├── main.py                    # สคริปต์หลักรัน Pipeline การเทรนโมเดล VGG 50 Epochs และบันทึกผลลัพธ์
│   ├── test_vgg.py                # สคริปต์สุ่มรูปภาพ 4 ภาพมาทดสอบพยากรณ์และแสดงผลกริด 2x2 พร้อม Confidence Score
│   │
│   └── outputs/                   # โฟลเดอร์เก็บผลลัพธ์และโมเดลที่บันทึก
│       ├── classes.json           # รายชื่อคลาส ['paper', 'rock', 'scissors']
│       ├── features.npy           # ข้อมูลรูปภาพทั้งหมดในรูป NumPy Array (uint8)
│       ├── labels.npy             # ข้อมูล Labels ทั้งหมด
│       ├── X_train.npy / X_val.npy / X_test.npy # ข้อมูลแบ่งชุดรูปภาพ
│       ├── y_train.npy / y_val.npy / y_test.npy # ข้อมูลแบ่งชุด Label
│       ├── vgg_model.keras        # โมเดล VGG DCNN ที่ฝึกสอนเสร็จสมบูรณ์
│       ├── history.json           # ค่าประวัติการฝึกสอน (Accuracy & Loss แต่ละ Epoch)
│       ├── confusion_matrix.png   # ภาพแผนภูมิความร้อน Confusion Matrix บนชุดข้อมูล Test Set
│       ├── training_history.png   # กราฟแสดงผล Training/Validation Accuracy และ Loss ตลอด 50 Epochs
│       └── prediction_sample.png  # ผลการสุ่มทดสอบพยากรณ์ภาพ 4 ตัวอย่างพร้อมค่าความมั่นใจ
│
└── README.md                      # เอกสารสรุปและอธิบายผลการทดลอง