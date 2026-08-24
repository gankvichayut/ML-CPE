from sklearn.svm import SVC

def train_svm(X_train, y_train, kernel='rbf'):
    # ไม่ต้องใช้ PCA สำหรับ Tabular Dataset
    model = SVC(kernel=kernel, C=1.0, random_state=42)
    model.fit(X_train, y_train)
    return model

def predict_svm(model, X_test):
    return model.predict(X_test)