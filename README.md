# Machine Learning Course (04-624-201)

This repository contains all laboratory assignments, source code, datasets, model implementations, and project documentation for the **Machine Learning (04-624-201)** course, Department of Computer Engineering, Faculty of Engineering, Rajamangala University of Technology Thanyaburi (RMUTT).

---

## Course Description
Fundamental concepts of machine learning, supervised and unsupervised learning, data preprocessing, feature engineering, classification, clustering, regression models, K-nearest neighbors (KNN), support vector machines (SVM), artificial neural networks (ANN/MLP), and evaluation metrics. The repository focuses on hands-on practical implementations using Python and modern data science / deep learning frameworks.

---

## Laboratory Directory & Overview

### 🔹 [LAB 02: Data Preprocessing](./LAB02)
* **Topic:** Data Exploration, Data Cleaning, and Feature Engineering
* **Dataset:** House Prices - Advanced Regression Techniques (`train.csv`)
* **Key Contents:**
  * Missing value imputation using statistical median and category labels
  * Handling duplicate records and assessing distribution skewness (SalePrice)
  * Feature encoding using Binary Mapping (`CentralAir`) and One-Hot Encoding (`HouseStyle`)
  * Correlation heatmap analysis to identify primary features for house price estimation

---

### 🔹 [LAB 03: Regression & Classification](./LAB03)
* **Topic:** Linear Regression vs. Logistic Regression & Dimensionality Reduction
* **Dataset:** UTKFace Face Analysis Dataset (`age_gender.csv`)
* **Key Contents:**
  * Pixel feature extraction ($48 \times 48 = 2,304$ dimensions) and standardization
  * Dimensionality reduction using **Principal Component Analysis (PCA)** to 50 Principal Components
  * Continuous age prediction using Simple vs. Multiple Linear Regression (MSE, MAE, $R^2$)
  * Binary gender classification using Logistic Regression with 2D Decision Boundary visualization and ROC-AUC evaluation

---

### 🔹 [LAB 04: K-Nearest Neighbors (KNN) & Clustering](./LAB04)
* **Topic:** Distance-Based Classification & Unsupervised Clustering
* **Dataset:** Heart Disease Diagnosis Dataset (`dataset_heart.csv`)
* **Key Contents:**
  * Feature standardization (Z-score Scaling) for distance preservation
  * KNN Classifier optimization across multiple neighbor parameters ($k \in [1, 21]$)
  * Unsupervised patient clustering using **K-Means Clustering** ($k=2$) via Elbow Method and Silhouette Analysis
  * Automated KNN cluster assignment for incoming patient data

---

### 🔹 [LAB 05: Support Vector Machine (SVM)](./LAB05)
* **Topic:** SVM Classification & Kernel Function Comparison
* **Dataset:** Breast Cancer Wisconsin Diagnostic Dataset (`brca.csv`)
* **Key Contents:**
  * End-to-end modular pipeline (`data_load`, `split_data`, `preprocess`, `svm_model`, `evaluate`)
  * Feature normalization using `StandardScaler`
  * Comparative performance benchmark among **Linear**, **Polynomial (Poly)**, and **Radial Basis Function (RBF)** kernels
  * Generation of Confusion Matrices and model serialization (`svm_model.pkl`) for clinical sample inference

---

### 🔹 [LAB 06: Neural Networks (MLP)](./LAB06)
* **Topic:** Deep Learning & Multilayer Perceptron for Computer Vision
* **Dataset:** Rock-Paper-Scissors Computer Vision Dataset (`rps-cv-images`)
* **Key Contents:**
  * Image ingestion, RGB conversion, and spatial resizing ($64 \times 64 \times 3$)
  * Multiclass Sequential MLP architecture with Batch Normalization and Dropout layers
  * Training stabilization with `Adam` optimizer, `EarlyStopping`, and `ReduceLROnPlateau` callbacks
  * Visualized training history curves (Loss & Accuracy) and random $2 \times 2$ grid test inference with confidence scores

---

## Tech Stack & Dependencies
* **Programming Language:** Python 3.13+
* **Data Processing & Math:** `pandas`, `numpy`
* **Machine Learning & Preprocessing:** `scikit-learn`
* **Deep Learning Framework:** `tensorflow` / `keras`
* **Computer Vision:** `opencv-python` (`cv2`)
* **Visualization:** `matplotlib`, `seaborn`

---

## Author Information
* **Name:** นายวิชยุตม์ แก้ววิเศษ (Mr. Vichayut Kaewwiset)
* **Student ID:** 116710400582-8
* **Department:** ภาควิชาวิศวกรรมคอมพิวเตอร์ (Department of Computer Engineering)
* **Faculty:** คณะวิศวกรรมศาสตร์ (Faculty of Engineering)
* **University:** มหาวิทยาลัยเทคโนโลยีราชมงคลธัญบุรี (RMUTT)
