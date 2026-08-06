# 📊 LAB 2: Data Preprocessing (House Prices Dataset)

โครงการนี้เป็นส่วนหนึ่งของวิชา **Machine Learning (04-624-201)**
ภาควิชาวิศวกรรมคอมพิวเตอร์ คณะวิศวกรรมศาสตร์ มหาวิทยาลัยเทคโนโลยีราชมงคลธัญบุรี

---

## 🎯 วัตถุประสงค์
1. เพื่อสำรวจและตรวจสอบคุณภาพของข้อมูล (Dataset Exploration)
2. เพื่อทำความสะอาดข้อมูล (Data Cleaning) เช่น การจัดการ Missing Values และ Duplicates
3. เพื่อทำ Data Transformation และ Feature Encoding เตรียมพร้อมสำหรับการพัฒนาแบบจำลอง ML

---

## 📁 โครงสร้างไฟล์ใน Repository
├── data/
│   ├── train.csv              # ชุดข้อมูลสำหรับฝึกฝน (Dataset หลัก)
│   └── data_description.txt   # คำอธิบายคอลัมน์ของข้อมูล
├── lab2_data_preprocessing.ipynb # Jupyter Notebook แสดงกระบวนการและโค้ดทั้งหมด
└── README.md                  # เอกสารอธิบายโครงการ


---

## 🛠️ เครื่องมือและไลบรารีที่ใช้ (Technologies Used)
* **Python** 3.x
* **Pandas** - การจัดการและการแปลงข้อมูลตาราง
* **NumPy** - การคำนวณเชิงตัวเลข
* **Matplotlib & Seaborn** - การแสดงผลข้อมูลด้วยกราฟ (Data Visualization)

---

## ⚙️ ขั้นตอนการดำเนินงาน (Data Preprocessing Steps)

### 1. Dataset Exploration (การสำรวจข้อมูล)
* ตรวจสอบขนาดของข้อมูล (Shape), ชนิดข้อมูล (Data Types) และสถิติเบื้องต้น
* ตรวจสอบค่าที่หายไป (Missing Values) และข้อมูลซ้ำ (Duplicate Records)

### 2. Data Visualization (การแสดงผลด้วยกราฟ)
* สร้าง **Histogram** ดูการกระจายตัวของราคาบ้าน (`SalePrice`)
* สร้าง **Correlation Heatmap** ดูความสัมพันธ์ระหว่างตัวแปรเชิงตัวเลข

### 3. Data Cleaning (การทำความสะอาดข้อมูล)
* **Missing Values Handling:** เติมค่าว่างคอลัมน์เชิงปริมาณด้วยมัธยฐาน (Median) และเติมข้อความ `'None'` สำหรับคอลัมน์เชิงกลุ่ม
* **Duplicate Removal:** ลบข้อมูลแถวที่ซ้ำกันออก
* **Data Type Conversion:** ปรับแต่งชนิดข้อมูลให้เหมาะสมสำหรับการวิเคราะห์

### 4. Feature Engineering
* **Label Encoding:** แปลงตัวแปรเชิงคุณภาพที่มีลำดับ (เช่น ค่า `CentralAir`) ให้เป็นตัวเลข 0 และ 1
* **One-Hot Encoding:** แปลงตัวแปรเชิงกลุ่ม (เช่น `HouseStyle`) ให้เป็นคอลัมน์ตัวเลข Binary (0/1)

---

## 📈 ผลลัพธ์และข้อสรุป (Results & Insights)
* **ราคาบ้าน (`SalePrice`):** มีลักษณะการกระจายตัวแบบเบ้ขวา (Right-skewed)
* **ความสัมพันธ์ของข้อมูล:** ตัวแปรที่มีความสัมพันธ์เชิงบวกสูงสุดกับราคาบ้าน ได้แก่ คุณภาพโดยรวมของบ้าน (`OverallQual`) และพื้นที่ใช้สอย (`GrLivArea`)
* ข้อมูลผ่านการ Clean และ Encode เรียบร้อยพร้อมนำไปใช้เทรนแบบจำลอง Machine Learning ต่อไป

---

## 👤 จัดทำโดย
* **ชื่อ-นามสกุล:** [ใส่ชื่อของคุณ]
* **รหัสนักศึกษา:** [ใส่รหัสนักศึกษา]
* **สาขา:** วิศวกรรมคอมพิวเตอร์
