# ML-09-Object Detection I

โปรเจกต์นี้เป็นการศึกษาและทดลองสถาปัตยกรรม **R-CNN และ Fast R-CNN** โดยใช้ภาษา Python และ TensorFlow/Keras ร่วมกับชุดข้อมูล Berkeley DeepDrive (BDD100K) ครอบคลุมตั้งแต่กระบวนการฝึกสอนโมเดล (Model Training), การประเมินผล (Evaluation), การทดสอบทำนายผลกับรูปภาพ (Image Inference) ไปจนถึงการทดสอบแบบเรียลไทม์ผ่านกล้องเว็บแคม (Real-time Webcam Testing) ผ่านการจำลองโครงสร้างเพื่อการศึกษา

---

## 1. ภาพรวมโปรเจกต์ (Project Overview)

โปรเจกต์นี้นำเสนอสถาปัตยกรรมโครงข่ายตรวจจับวัตถุ (Object Detection) บนฐานของ CNN โดยใช้ชุดข้อมูล BDD100K ร่วมกัน แต่ละสถาปัตยกรรมถูกจัดสรรแยกออกเป็นแต่ละโฟลเดอร์ พร้อมด้วยสคริปต์สำหรับการเทรน การประเมินผล และการทดสอบที่เป็นอิสระจากกัน

**จุดเด่นและฟีเจอร์:**
- การศึกษาและสำรวจสถาปัตยกรรมบนพื้นฐานของ R-CNN
- ใช้ชุดข้อมูล BDD100K ร่วมกัน ครอบคลุมวัตถุบนท้องถนน 10 คลาส
- มีระบบฝึกสอนและประเมินประสิทธิภาพของโมเดล
- รองรับการทดสอบพยากรณ์ภาพนิ่งและการทดสอบสดผ่านกล้องเว็บแคม
- จัดเก็บ Model Checkpoints และผลลัพธ์การทดลองอย่างเป็นระบบ
- แยกโค้ดแต่ละสถาปัตยกรรมอย่างชัดเจนเพื่อความสะดวกในการเปรียบเทียบเชิงโครงสร้าง

---

## 2. ข้อมูลชุดข้อมูล (Dataset)

โปรเจกต์นี้ใช้ชุดข้อมูล [BDD100K Dataset](https://www.kaggle.com/datasets/a7madmostafa/bdd100k-yolo?resource=download) ในรูปแบบ YOLO Annotation Format

| ชุดข้อมูล (Split) | จำนวนภาพ (Images) |
|---|---:|
| ชุดฝึกสอน (Training) | 70,000 |
| ชุดตรวจสอบ (Validation) | 10,000 |
| ชุดทดสอบ (Test) | มีให้ในชุดข้อมูล (ไม่ได้ระบุจำนวนแน่ชัด) |

**คลาสวัตถุ (Classes):** `person`, `rider`, `car`, `bus`, `truck`, `bike`, `motor`, `traffic light`, `traffic sign`, `train`

ชุดข้อมูลนี้ถูกใช้งานร่วมกันในทุกสถาปัตยกรรม โดยการระบุตำแหน่งแบบ YOLO สามารถแปลงเป็นพิกัด Bounding Box ตามที่โมเดลต้องการได้ในขั้นตอน Preprocessing

---

## 3. โครงสร้างโปรเจกต์ (Project Structure)

```text
ML-09-Object20l/
│
├── 01-R-CNN/
│   ├── outputs/
│   │   └── image_result.jpg
│   ├── _core.py
│   ├── evaluate.py
│   ├── image_test.py
│   ├── realtime_test.py
│   └── train.py
│
├── 02-Fast-R-CNN/
│   ├── outputs/
│   │   ├── evaluation.json
│   │   └── image_result.jpg
│   ├── evaluate.py
│   ├── image_test.py
│   ├── realtime_test.py
│   └── train.py
│
├── dataset/
│   ├── test/
│   │   ├── images/
│   │   └── labels/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   └── data.yaml
│
├── README.md
└── requirements.txt