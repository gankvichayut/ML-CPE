import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def plot_models_comparison(all_histories, save_path):
    """
    วาดกราฟเปรียบเทียบ Validation Accuracy ที่มีความนิ่งและอ่านค่าง่าย
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#8A2BE2", "#D9534F", "#5CB85C", "#337AB7", "#F0AD4E", "#5BC0DE"]
    markers = ["o-", "s-", "^-", "d-", "v-", "x-"]

    for idx, (name, history) in enumerate(all_histories.items()):
        val_acc = [acc * 100 for acc in history.history["val_accuracy"]]
        # ใช้ค่าเฉลี่ย 3 epoch สุดท้ายเพื่อแสดงความแม่นยำจริงที่โมเดลนิ่งแล้ว
        final_acc = np.mean(val_acc[-3:])
        
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]

        ax.plot(
            range(1, len(val_acc) + 1),
            val_acc,
            marker,
            color=color,
            markevery=max(1, len(val_acc) // 10),
            linewidth=2,
            alpha=0.85,
            label=f"{name}: {final_acc:.2f}%"
        )

    ax.set_title("Training Performance", fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Epochs", fontsize=12)
    ax.set_ylabel("Validation Accuracy (%)", fontsize=12)
    ax.set_ylim(20, 105)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{int(y)}%"))
    ax.legend(loc="lower right", frameon=True, fontsize=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"\n[Saved Comparison Plot]: {save_path}")


def evaluate_model(y_test, predictions, classes, save_path=None):
    labels = list(range(len(classes)))
    accuracy = accuracy_score(y_test, predictions)

    print("\n------------ Final Evaluation ------------------")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, labels=labels, target_names=classes, zero_division=0))

    if save_path:
        matrix = confusion_matrix(y_test, predictions, labels=labels)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(np.arange(len(classes)), classes)
        ax.set_yticks(np.arange(len(classes)), classes)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        threshold = matrix.max() / 2
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, matrix[i, j], ha="center", va="center",
                        color="white" if matrix[i, j] > threshold else "black")
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return accuracy