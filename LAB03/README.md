# LAB 3: Regression & Classification (Face Analysis Dataset)

This project is part of the **Machine Learning (04-624-201)** course  
Department of Computer Engineering, Faculty of Engineering, Rajamangala University of Technology Thanyaburi

---

# Data

Kaggle Dataset: https://www.kaggle.com/datasets/nipunarora8/age-gender-and-ethnicity-face-data-csv

---

## Objectives
1. To enable students to understand the principles of Regression and Classification, which are fundamental techniques in Supervised Learning, and to clearly explain the difference between continuous value prediction and categorical data classification.
2. To enable students to prepare data for model development by selecting appropriate features and applying Principal Component Analysis (PCA) to reduce dimensionality while enhancing learning efficiency.
3. To enable students to develop Linear Regression models for age prediction from facial images and Classification models for gender classification from facial images, along with comparing the outcomes of each method.
4. To enable students to write Python programs and leverage Machine Learning libraries to build, train, test, and evaluate model performance.
5. To enable students to analyze and interpret model evaluation metrics such as MSE, MAE, R² Score, Accuracy, Precision, Recall, F1-score, ROC Curve, and AUC, as well as present their findings via GitHub.

---

## Repository Structure
```text
ML-03-Regression-Classification/
│
├── age_gender.csv          # UTKFace face image dataset (Features & Labels)
├── LAB3_code.ipynb         # Main Jupyter Notebook containing all code and experiment results
└── README.md               # Project documentation and experiment summary
```

---

## Results & Insights
* **Dimensionality Reduction with PCA:** Reducing feature dimensions from 2,304 pixels to 50 Principal Components effectively preserved the dominant variance of facial structures while suppressing background noise.
* **Regression Performance (Age Prediction):** Multiple Linear Regression achieved superior accuracy compared to Simple Linear Regression, as it incorporated multidimensional facial structural features simultaneously.
* **Classification Performance (Gender Classification):** Logistic Regression successfully established clear Decision Boundaries to classify gender, yielding satisfactory ROC-AUC and F1-Score metrics.
* **Differences between Regression and Classification:**
  * **Regression:** Focuses on minimizing quantitative continuous error ($y \in \mathbb{R}$) and is evaluated using residual metrics (MSE, MAE, $R^2$).
  * **Classification:** Focuses on finding decision rules for class probabilities ($P(y=1)$) and is evaluated using Confusion Matrix, Precision, Recall, F1-Score, and ROC-AUC.

---

## Prepared by
* **Full Name:** Vichayut Kaewwiset
* **Student ID:** 116710400582-8
* **Department:** Computer Engineering
