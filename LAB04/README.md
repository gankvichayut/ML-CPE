# LAB 4: K-Nearest Neighbors (KNN) & Clustering (Heart Disease Dataset)

This project is part of the **Machine Learning (04-624-201)** course  
Department of Computer Engineering, Faculty of Engineering, Rajamangala University of Technology Thanyaburi

---

# Data

Kaggle Dataset: https://www.kaggle.com/datasets/utkarshx27/heart-disease-diagnosis-dataset

---

## Objectives
1. To apply the K-Nearest Neighbors (KNN) algorithm for medical data classification (Classification).
2. To compare model performance when tuning the number of neighbors (k values).
3. To group patient data using K-Means Clustering evaluated via the Elbow Method and Silhouette Score (Clustering).
4. To study the importance of Feature Standardization on distance-based algorithms.

---

## Repository Structure
```text
LAB04/
│
├── classification/
│   ├── dataset_heart.csv          # Heart disease dataset for Classification tasks
│   ├── main.py                    # Main script for training and evaluating KNN
│   ├── data_loader.py             # Loads data, splits Train/Val/Test, and performs Standardization
│   ├── knn_tf.py                  # KNN model class built on TensorFlow
│   ├── evaluate.py                # Evaluation functions, k-curve plotting, and Confusion Matrix
│   └── outputs/
│       ├── 01_k_curve.png         # Validation Accuracy vs. k-value plot
│       ├── 02_confusion_matrix.png # Confusion Matrix on the Test Set
│       └── predictions.csv        # Prediction results compared against ground truth
│
├── clustering/
│   ├── dataset_heart.csv          # Heart disease dataset for Clustering tasks
│   ├── main.py                    # Main script for running K-Means and KNN cluster assignment
│   ├── data_loader.py             # Loads data and applies Standardization to features only
│   ├── kmeans_tf.py               # K-Means Clustering class built on TensorFlow
│   ├── knn_tools.py               # KNN class for assigning clusters to new incoming data
│   ├── visualize.py               # Visualization functions for Elbow Curve and Scatter Plots
│   └── outputs/
│       ├── 01_elbow.png           # Elbow Method curve for optimal k selection
│       ├── 02_clusters.png        # Cluster scatter plot (Age vs. Max Heart Rate)
│       ├── cluster_summary.csv    # Statistical summary of mean values per cluster
│       └── clustered_heart_data.csv # Dataset with assigned cluster labels for each patient
│
└── README.md                      # Project documentation and experiment summary
```

---

## Results & Insights
* **KNN Classification Performance:** The model achieved a Test Set Accuracy of **79.63%**, significantly outperforming majority-class baseline guessing.
* **K-Means Clustering Analysis:**
  * **Cluster 0 (Low Risk):** Average age of 52 years, mean maximum heart rate of 159.9 bpm, exercise-induced angina in only 11.9% of cases (corresponding to 81.5% actual negative diagnoses).
  * **Cluster 1 (High Risk):** Average age of 58.4 years, mean maximum heart rate of 132.8 bpm, exercise-induced angina in 67.6% of cases (corresponding to 87.3% actual positive diagnoses).
* **Importance of Feature Standardization:** Because medical variables vary widely in units and numerical ranges, standardization prevents high-magnitude features from dominating Euclidean distance calculations.

---

## Prepared by
* **Full Name:** Vichayut Kaewwiset
* **Student ID:** 116710400582-8
* **Department:** Computer Engineering
