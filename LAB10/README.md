# ML-10-Object Detection II

โปรเจกต์นี้เป็นการนำสถาปัตยกรรม **YOLOv8 สำหรับการตรวจจับวัตถุ (Object Detection)** มาประยุกต์ใช้งานร่วมกับชุดข้อมูล Berkeley DeepDrive (BDD100K) เพื่อตรวจจับวัตถุบนท้องถนน เช่น ยานพาหนะ, คนเดินเท้า และป้ายสัญลักษณ์จราจร ครอบคลุมตั้งแต่การฝึกสอนโมเดล (Fine-tuning), การประเมินผลประสิทธิภาพ (Evaluation), การทดสอบพยากรณ์ภาพนิ่ง (Image Inference) ไปจนถึงการตรวจจับแบบสดผ่านกล้องเว็บแคม (Real-time Detection)

---

## 1. ภาพรวมโปรเจกต์ (Project Overview)

โปรเจกต์นี้ทำการ Fine-tune โครงข่าย **Ultralytics YOLOv8n** ด้วยน้ำหนักโมเดลเริ่มต้น (Pretrained Weights) บนชุดข้อมูล BDD100K

**จุดเด่นและฟีเจอร์:**
- ตรวจจับวัตถุบนท้องถนนครอบคลุมทั้งหมด 10 คลาส
- พัฒนาต่อยอดด้วยการ Fine-tune บนโครงสร้าง YOLOv8 แบบ Pretrained
- ประเมินผลความแม่นยำด้วยค่า Precision, Recall, mAP50 และ mAP50-95
- รองรับการทดสอบพยากรณ์ภาพนิ่งและการตรวจจับแบบเรียลไทม์ผ่านกล้องเว็บแคม
- มีระบบบันทึก Model Checkpoints และเก็บบันทึกประวัติการเทรนแต่ละ Epoch ในรูปแบบไฟล์ CSV ให้อัตโนมัติ

---

## 2. ข้อมูลชุดข้อมูล (Dataset)

โปรเจกต์นี้ใช้ชุดข้อมูล [BDD100K Dataset](https://www.kaggle.com/datasets/a7madmostafa/bdd100k-yolo?resource=download) ในรูปแบบ YOLO Annotation Format

| ชุดข้อมูล (Split) | จำนวนภาพ (Images) |
|---|---:|
| ชุดฝึกสอน (Training) | 70,000 |
| ชุดตรวจสอบ (Validation) | 10,000 |
| ชุดทดสอบ (Test) | มีให้ในชุดข้อมูล (Provided in dataset) |

**คลาสวัตถุ (Classes):** `person`, `rider`, `car`, `bus`, `truck`, `bike`, `motor`, `traffic light`, `traffic sign`, `train`

แต่ละบรรทัดในไฟล์ Label แบบ YOLO จะประกอบด้วย Class ID และพิกัด Bounding Box ในรูปแบบ Normalized Coordinate (x_center, y_center, width, height)

---

## 3. โครงสร้างโปรเจกต์ (Project Structure)

```text
ML-10-Object20II/
├── dataset/
│   ├── test/{images,labels}/
│   ├── train/{images,labels}/
│   ├── val/{images,labels}/
│   └── data.yaml
│
├── runs/
│   ├── bdd100k/
│   │   ├── weights/
│   │   │   ├── best.pt
│   │   │   └── last.pt
│   │   └── results.csv
│   ├── detect/
│   ├── evaluation/
│   └── image_test/
│
├── weights/
│   └── yolo26n.pt
│
├── evaluate.py
├── image_test.py
├── realtime_test.py
├── train.py
├── yolov8n.pt
├── README.md
└── requirements.txt