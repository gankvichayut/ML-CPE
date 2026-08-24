# ML-5-Support Vector Machine (SVM)

Build a simple SVM pipeline using Python, including image data loading, preprocessing, feature scaling, model training, evaluation, and prediction.

# Data
on-dataset
Kaggle House Prices - Advanced Regression Techniques Dataset: 
   https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
   
# Structure

```text
ML-05-SVM/
│
├── PetImages/
│   ├── Cat/
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   └── ...
│   │
│   └── Dog/
│       ├── 0.jpg
│       ├── 1.jpg
│       └── ...
│
├── classification/
│   ├── main.py
│   ├── test_svm.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── split_data.py
│   ├── svm_model.py
│   ├── evaluate.py
│   └── outputs/
│       ├── features.npy
│       ├── labels.npy
│       ├── classes.json
│       ├── X_train.npy
│       ├── X_test.npy
│       ├── y_train.npy
│       ├── y_test.npy
│       ├── scaler.pkl
│       ├── svm_model.pkl
│       └── confusion_matrix.png
│
├── requirements.txt
└── link-data.txt
```

