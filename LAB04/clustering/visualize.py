"""
visualize.py
หน้าที่: สร้างกราฟแสดงผลลัพธ์การจัดกลุ่มและบันทึกลงไฟล์รูปภาพ
"""

import matplotlib
matplotlib.use("Agg")  # สำหรับ Environment ที่ไม่มี GUI Display

import matplotlib.pyplot as plt

def plot_elbow(k_values, inertias, out_path):
    """วาดกราฟ Elbow Method เพื่อดูจุดหักงอของค่า Inertia"""
    plt.figure(figsize=(7, 4.5))
    plt.plot(k_values, inertias, "o-", color="#2ca02c", linewidth=2, markersize=6)
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia (Sum of Squared Distances)")
    plt.title("Elbow Method for Optimal k")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()

def plot_clusters(X_raw_pair, labels, out_path, x_name="Feature 1", y_name="Feature 2"):
    """วาด Scatter Plot แสดงการกระจายตัวของสมาชิกในแต่ละ Cluster"""
    plt.figure(figsize=(7.5, 6))

    for c in range(labels.max() + 1):
        members = labels == c
        plt.scatter(
            X_raw_pair[members, 0], 
            X_raw_pair[members, 1],
            s=35, 
            alpha=0.7, 
            label=f"Cluster {c}"
        )

    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title("K-Means Clustering Distribution")
    plt.legend(fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()