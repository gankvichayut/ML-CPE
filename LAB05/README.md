# LAB 5: Support Vector Machine (SVM) on Breast Cancer Dataset

This project is part of the **Machine Learning (04-624-201)** course  
Department of Computer Engineering, Faculty of Engineering, Rajamangala University of Technology Thanyaburi

---

# Data

Kaggle Dataset: https://www.kaggle.com/datasets/utkarshx27/breast-cancer-wisconsin-diagnostic-dataset

---

## Objectives
1. To enable students to understand the principles of Support Vector Machine (SVM) and its practical applications in both classification and prediction tasks.
2. To enable students to develop workflow diagrams, process data, and visualize outcomes to build working prototype systems using SVM for accurate classification and forecasting.
3. To apply Support Vector Machine (SVM) to medical data classification using the Breast Cancer Wisconsin Diagnostic Dataset (`brca.csv`).
4. To study and compare the performance of SVM models across various kernel functions, namely Linear, Polynomial (Poly), and Radial Basis Function (RBF).
5. To enable students to write Python programs and evaluate model performance using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix, as well as present their project via GitHub.

---

## Repository Structure
```text
LAB05/
│
├── dataset/
│   └── brca.csv                   # Breast Cancer Wisconsin (Diagnostic) dataset
│
├── mini-project/
│   ├── main.py                    # Main script to run pipeline, train, and compare kernels
│   ├── test_svm.py                # Script for random sample testing and prediction using the best model
│   ├── data_load.py               # Loads data, encodes labels (Benign=0, Malignant=1), and extracts features
│   ├── preprocess.py              # Scales feature data using StandardScaler
│   ├── split_data.py              # Splits dataset into Train 80% and Test 20% with stratification
│   ├── svm_model.py               # Builds and trains Support Vector Classifier (SVC) models
│   ├── evaluate.py                # Evaluates performance and generates Confusion Matrix plots
│   └── outputs/
│       ├── classes.json           # Target class names ['Benign', 'Malignant']
│       ├── X_train.npy            # Scaled feature matrix for training set
│       ├── X_test.npy             # Scaled feature matrix for test set
│       ├── y_train.npy            # Target label array for training set
│       ├── y_test.npy             # Target label array for test set
│       ├── scaler.pkl             # Fitted StandardScaler object
│       ├── svm_model.pkl          # Serialized best-performing SVM model
│       ├── confusion_matrix_linear.png # Confusion matrix plot for Linear Kernel
│       ├── confusion_matrix_poly.png   # Confusion matrix plot for Polynomial Kernel
│       └── confusion_matrix_rbf.png    # Confusion matrix plot for RBF Kernel
│
└── README.md                      # Experiment summary and project documentation
```

---

## Results & Insights
* **Importance of Feature Standardization:** Because SVM models operate by calculating the geometric margin between support vectors and the decision hyperplane, standardizing medical features to a common scale is a crucial step that enhances accuracy and eliminates model bias.
* **SVM Kernel Performance:**
  * Both **Linear Kernel** and **RBF Kernel** effectively constructed decision boundaries capable of cleanly separating malignant and benign cases with high performance.
  * Choosing a kernel function tailored to the underlying geometric distribution of medical features directly influences classification accuracy and the clinical recall rate for malignant tumor detection.
* **Practical Application:** The trained SVM pipeline can be serialized and deployed as an automated diagnostic assistance prototype to classify new clinical patient samples accurately and efficiently.

---

## Prepared by
* **Full Name:** Vichayut Kaewwiset
* **Student ID:** 116710400582-8
* **Department:** Computer Engineering
