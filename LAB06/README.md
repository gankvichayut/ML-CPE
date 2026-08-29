# LAB 6: Neural Network (MLP) for Image Recognition (Rock-Paper-Scissors Dataset)

This project is part of the **Machine Learning (04-624-201)** course  
Department of Computer Engineering, Faculty of Engineering, Rajamangala University of Technology Thanyaburi

---

# Data

Kaggle Dataset: https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors/

---

## Objectives
1. To enable students to understand the architecture and working principles of Artificial Neural Networks (Multilayer Perceptron: MLP) in image classification tasks.
2. To practice computer vision data preparation and preprocessing pipelines, including image resizing, color space conversion (BGR to RGB), and tensor structuring for deep learning models.
3. To construct and train neural network models using TensorFlow / Keras by implementing Batch Normalization, Dropout layers for regularization, and training callbacks (Early Stopping and ReduceLROnPlateau).
4. To analyze learning curves (Loss & Accuracy) across both Training and Validation sets to evaluate overfitting and underfitting behaviors.
5. To evaluate model performance using Accuracy, Classification Report (Precision, Recall, F1-Score), and Confusion Matrix, and to perform sample image inference with confidence score visualization.

---

## Repository Structure
```text
LAB06/
│
├── rps-cv-images/                 # Raw image dataset directory
│   ├── paper/                     # Paper class images
│   ├── rock/                      # Rock class images
│   ├── scissors/                  # Scissors class images
│   └── README_rpc-cv-images.txt   # Dataset description file
│
├── mini-project/
│   ├── data_loader.py             # Script to load images from directories and convert to arrays
│   ├── preprocessing.py           # Functions for image resizing (64x64) and RGB conversion
│   ├── split_data.py              # Splits dataset into Train (70%), Validation (10%), and Test (20%)
│   ├── nn_model.py                # Sequential MLP model architecture, rescaling, and training logic
│   ├── evaluate.py                # Evaluation metrics, Confusion Matrix, and training history plotting
│   ├── main.py                    # Main pipeline execution and model serialization script
│   ├── test_nn.py                 # Script to randomly sample 4 test images and display 2x2 prediction grid
│   │
│   └── outputs/                   # Directory containing saved outputs and model artifacts
│       ├── classes.json           # Target class names ['paper', 'rock', 'scissors']
│       ├── history.json           # Per-epoch training loss and accuracy logs
│       ├── features.npy           # Full feature array in NumPy format
│       ├── labels.npy             # Full label array
│       ├── X_train.npy / X_val.npy / X_test.npy # Partitioned image arrays
│       ├── y_train.npy / y_val.npy / y_test.npy # Partitioned target label arrays
│       ├── nn_model.keras         # Serialized Neural Network model in Keras format
│       ├── confusion_matrix.png   # Confusion Matrix heatmap image
│       ├── training_history.png   # Loss and Accuracy curves across training and validation
│       └── prediction_sample.png  # Visualized 2x2 sample prediction output
│
└── README.md                      # Experiment summary and project documentation
```

---

## Results & Insights
* **Regularization Effectiveness:** Incorporating Batch Normalization and Dropout layers successfully prevented the high-dimensional Multilayer Perceptron ($12,288$ features) from overfitting on the training set.
* **Role of Validation Set & Early Stopping:** Continuous validation monitoring paired with Early Stopping ensured that training halted at the optimal generalization point while restoring the best model weights.
* **Hand Gesture Classification Accuracy:** The Fully-Connected Neural Network accurately learned discriminative visual patterns among the three gesture classes, delivering high confidence predictions during test set inference.

---

## Prepared by
* **Full Name:** Vichayut Kaewwiset
* **Student ID:** 116710400582-8
* **Department:** Computer Engineering
