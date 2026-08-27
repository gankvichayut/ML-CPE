"""
knn_tf.py
หน้าที่: สร้างคลาส TFKNNClassifier คำนวณระยะทางและโหวตคลาสโดยใช้ TensorFlow
"""

import numpy as np
import tensorflow as tf

class TFKNNClassifier:
    def __init__(self, k=5):
        self.k = k  # จำนวนเพื่อนบ้านที่ใช้ในการตัดสินใจ

    def fit(self, X, y):
        """บันทึกข้อมูล Train เข้าสู่ Memory ในรูปแบบ Tensor"""
        self.X_train = tf.constant(X, dtype=tf.float32)
        self.y_train = tf.constant(y, dtype=tf.int32)
        self.n_classes = int(y.max()) + 1
        return self

    def _distance(self, X_new):
        """
        คำนวณ Euclidean Distance ระหว่างชุดข้อมูลใหม่ กับ ข้อมูล Train ทุกตัว
        Formula: d(x, y) = sqrt( sum( (x_i - y_i)^2 ) )
        """
        # ใช้ Broadcasting สร้าง 3D Tensor: (n_new, n_train, n_features)
        diff = X_new[:, None, :] - self.X_train[None, :, :]
        return tf.sqrt(tf.reduce_sum(tf.square(diff), axis=2))

    def predict(self, X):
        """ทำนายคลาสของชุดข้อมูลใหม่"""
        X = tf.constant(X, dtype=tf.float32)

        # คำนวณระยะทาง
        dist = self._distance(X)

        # หา k เพื่อนบ้านที่ใกล้ที่สุด (ค่าติดลบของ distance ที่มากสุด = ระยะทางน้อยสุด)
        _, idx = tf.math.top_k(-dist, k=self.k)
        neighbor_labels = tf.gather(self.y_train, idx)

        # โหวต Majority Vote ผ่าน One-Hot Encoding
        onehot = tf.one_hot(neighbor_labels, depth=self.n_classes)
        votes = tf.reduce_sum(onehot, axis=1)

        # เลือกคลาสที่มีผลรวมคะแนนโหวตสูงสุด
        return tf.argmax(votes, axis=1).numpy()

    def score(self, X, y):
        """คำนวณค่า Accuracy"""
        return float(np.mean(self.predict(X) == y))