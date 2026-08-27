"""
knn_tools.py
หน้าที่:
1. นำอัลกอริทึม KNN มาประยุกต์ใช้กับผลลัพธ์ของ Clustering
2. รับข้อมูลใหม่เข้ามาแล้วจำแนกเข้ากลุ่ม (Cluster) โดยไม่ต้องรัน K-Means ใหม่ทั้งหมด
"""

import numpy as np
import tensorflow as tf


class KNNClusterAssigner:
    def __init__(self, k=5):
        """
        - k: จำนวนเพื่อนบ้านที่ใช้ในการพิจารณาจัดกลุ่ม
        """
        self.k = k

    def fit(self, X, cluster_labels):
        """
        บันทึกพิกัดข้อมูลเดิม และรหัส Cluster ที่ถูกจัดไว้แล้วลงใน Tensor Memory
        - X             : ชุดข้อมูลที่ทราบ Cluster แล้ว
        - cluster_labels: รหัส Cluster ของแต่ละแถว (0, 1, ..., n_clusters-1)
        """
        self.X = tf.constant(X, dtype=tf.float32)
        self.labels = tf.constant(cluster_labels, dtype=tf.int32)
        self.n_clusters = int(cluster_labels.max()) + 1
        return self

    def predict(self, X_new):
        """
        ทำนายว่าข้อมูลจุดใหม่ (X_new) ควรถูกจัดให้อยู่ใน Cluster ใด
        """
        X_new = tf.constant(X_new, dtype=tf.float32)

        # 1. วัดระยะทาง Euclidean Distance จากจุดใหม่ทุกจุด ไปยังจุดข้อมูลเดิมทุกจุด
        diff = X_new[:, None, :] - self.X[None, :, :]
        dist = tf.sqrt(tf.reduce_sum(tf.square(diff), axis=2))

        # 2. ค้นหา k จุดเพื่อนบ้านที่ใกล้ที่สุด (ใส่เครื่องหมายลบ เพื่อให้ top_k หาค่าระยะทางที่น้อยที่สุด)
        _, idx = tf.math.top_k(-dist, k=self.k)
        neighbor_labels = tf.gather(self.labels, idx)

        # 3. รวมคะแนนโหวต (Majority Vote) ผ่าน One-Hot Encoding
        onehot = tf.one_hot(neighbor_labels, depth=self.n_clusters)
        votes = tf.reduce_sum(onehot, axis=1)

        # 4. ส่งคืนกลุ่มที่ได้รับคะแนนโหวตสูงสุด
        return tf.argmax(votes, axis=1).numpy().astype("int32")