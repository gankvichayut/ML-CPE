"""
kmeans_tf.py
หน้าที่:
1. ทำ K-Means Clustering โดยคำนวณแบบ Matrix Operation บน TensorFlow
2. อัปเดตตำแหน่งจุดศูนย์กลาง (Centroids) แบบ Iterative จนกว่าจะคงที่ (Convergence)
3. คำนวณค่า Inertia (Sum of Squared Errors) สำหรับนำไปใช้วิเคราะห์ Elbow Method
"""

import numpy as np
import tensorflow as tf


class TFKMeans:
    def __init__(self, n_clusters=4, max_iter=100, seed=42):
        """
        กำหนด Hyperparameters ของโมเดล K-Means
        - n_clusters: จำนวนกลุ่มที่ต้องการแบ่ง (ค่า k)
        - max_iter  : จำนวนรอบการวนซ้ำสูงสุด
        - seed      : กำหนด Random State เพื่อให้ผลลัพธ์สุ่ม Centroid เริ่มต้นเหมือนเดิมทุกครั้ง
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.seed = seed

    def _distance(self, X, centroids):
        """
        คำนวณ Euclidean Distance ระหว่างทุกจุดข้อมูล กับ จุดศูนย์กลาง (Centroids) ทุกจุด
        - X shape        : (n_samples, n_features)
        - centroids shape: (n_clusters, n_features)
        - ผลลัพธ์ shape  : (n_samples, n_clusters)
        """
        # ใช้ Tensor Broadcasting ขยายมิติเพื่อหาผลต่างทุกคู่: (n_samples, n_clusters, n_features)
        diff = X[:, None, :] - centroids[None, :, :]
        # คำนวณ sqrt( sum( (x - c)^2 ) ) ตามแกน Feature (axis=2)
        return tf.sqrt(tf.reduce_sum(tf.square(diff), axis=2))

    def fit(self, X):
        """
        เทรนโมเดล K-Means จนกว่าตำแหน่ง Centroids จะหยุดนิ่ง หรือครบจำนวนรอบสูงสุด
        """
        X = tf.constant(X, dtype=tf.float32)
        n_samples = X.shape[0]

        # STEP 0: สุ่มเลือกตัวอย่างข้อมูลมา k จุด เพื่อใช้เป็น Centroids เริ่มต้น
        rng = np.random.default_rng(self.seed)
        start_idx = rng.choice(n_samples, size=self.n_clusters, replace=False)
        centroids = tf.gather(X, start_idx)

        # วนซ้ำขั้นตอน Expectation-Maximization (Assign -> Update)
        for step in range(self.max_iter):
            # STEP 1 (ASSIGNMENT): คำนวณระยะห่าง แล้วกำหนดให้แต่ละจุดอยู่กับ Centroid ที่ใกล้ที่สุด (argmin)
            dist = self._distance(X, centroids)
            labels = tf.argmin(dist, axis=1, output_type=tf.int32)

            # STEP 2 (UPDATE): คำนวณค่าเฉลี่ย (Mean) ของสมาชิกในแต่ละกลุ่ม เพื่อเป็น Centroid พิกัดใหม่
            new_centroids = []
            for c in range(self.n_clusters):
                # ดึงเฉพาะจุดข้อมูลที่เป็นสมาชิกของ Cluster c
                members = tf.boolean_mask(X, labels == c)
                
                # หาก Cluster มีสมาชิก ให้หา Mean แต่ถ้าเป็น Empty Cluster ให้ใช้ Centroid เดิมป้องกัน Error
                if tf.shape(members)[0] > 0:
                    new_centroids.append(tf.reduce_mean(members, axis=0))
                else:
                    new_centroids.append(centroids[c])
            new_centroids = tf.stack(new_centroids)

            # ตรวจสอบการลู่เข้า (Convergence Check)
            # ถ้าระยะการเคลื่อนที่สูงสุดของ Centroid น้อยกว่า 1e-4 ถือว่าจุดศูนย์กลางนิ่งแล้ว
            moved = float(tf.reduce_max(tf.abs(new_centroids - centroids)))
            centroids = new_centroids
            if moved < 1e-4:
                break

        # บันทึกผลลัพธ์หลังโมเดลทำงานเสร็จสิ้น
        dist = self._distance(X, centroids)
        self.labels_ = tf.argmin(dist, axis=1, output_type=tf.int32).numpy()
        self.centroids_ = centroids.numpy()
        self.n_iter_ = step + 1

        # คำนวณค่า Inertia (ผลรวมระยะทางยกกำลังสองจากแต่ละจุดไปยัง Centroid ตัวเอง)
        # ใช้สำหรับวาดกราฟ Elbow Curve เพื่อเลือกค่า k ที่ดีที่สุด
        self.inertia_ = float(tf.reduce_sum(tf.square(tf.reduce_min(dist, axis=1))))
        return self

    def fit_predict(self, X):
        """เทรนโมเดลและส่งคืนค่า Cluster Label ของข้อมูลแต่ละแถว"""
        return self.fit(X).labels_