# LAB 2: Data Preprocessing (House Prices Dataset)

This project is part of the **Machine Learning (04-624-201)** course  
Department of Computer Engineering, Faculty of Engineering, Rajamangala University of Technology Thanyaburi

---

# Data

Kaggle Dataset: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/

---

## Objectives
1. To enable students to understand the principles and importance of Data Preprocessing, including the steps of inspecting, cleaning, transforming, and preparing high-quality data prior to applying Machine Learning.
2. To enable students to inspect and analyze data quality, such as checking for Missing Values, Duplicate Data, Outliers, and Inconsistent Data, along with selecting appropriate remediation methods.
3. To enable students to apply techniques such as Data Cleaning, Data Transformation, Data Encoding (Label Encoding & One-Hot Encoding), Feature Scaling, or Data Normalization, which serve as the foundation for preparing data into suitable formats for analysis and model development.
4. To enable students to program data preprocessing pipelines using Python and relevant libraries, verify and evaluate data quality before real-world deployment, and present their work via GitHub.

---

## Repository Structure
```text
LAB02/
│
├── house-prices-data/
│   ├── train.csv                  # Training dataset (Features & SalePrice)
│   ├── test.csv                   # Testing dataset
│   ├── data_description.txt       # Full descriptions of all feature columns
│   └── sample_submission.csv      # Sample submission format
│
├── LAB2_code.ipynb                # Main Jupyter Notebook containing all code and experiment results
└── README.md                      # Experiment summary and project documentation
```

---

## 📈 Results & Insights
* **Data Quality & Completeness:** Cleaning data and imputing missing values accurately according to the specific domain context of each feature prevents the loss of critical information and minimizes error during model development.
* **Feature Correlation with Sale Price:** Overall material and finish quality (`OverallQual`) and above-ground living area (`GrLivArea`) are the primary factors showing the strongest positive correlation with house prices (`SalePrice`).
* **Data Readiness for Machine Learning:** Datasets processed through comprehensive Data Cleaning, missing value imputation, and categorical encoding are well-structured and ready for train/test splitting and feeding into subsequent regression models.

---

## 👤 Prepared by
* **Full Name:** Wichayut Kaewwiset
* **Student ID:** 116710400582-8
* **Department:** Computer Engineering
