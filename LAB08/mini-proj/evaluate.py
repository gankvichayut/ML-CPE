import matplotlib

# Set backend before pyplot, so it works without a display
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


def evaluate_model(y_test, predictions, classes, save_path=None):

    # Pin label order so target_names always matches the columns
    labels = list(range(len(classes)))

    accuracy = accuracy_score(y_test, predictions)

    print("\n------------ Evaluation ------------------")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")

    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        target_names=classes,
        zero_division=0
    )

    print(report)
    print("Confusion Matrix:")

    matrix = confusion_matrix(y_test, predictions, labels=labels)
    print(matrix)

    if save_path:
        plot_confusion_matrix(matrix, classes, save_path)
        print(f"Saved: {save_path}")

    return accuracy


def plot_confusion_matrix(matrix, classes, save_path):

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(matrix, cmap="Blues")

    ax.set_xticks(np.arange(len(classes)), classes)
    ax.set_yticks(np.arange(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    threshold = matrix.max() / 2
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, matrix[i, j], ha="center", va="center",
                    color="white" if matrix[i, j] > threshold else "black")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_history(history, save_path):
    epochs = list(range(1, len(history.history["accuracy"]) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=200)

    # ---------------- กราฟ Accuracy (สไตล์คล้าย models_comparison แบบถี่ๆ) ----------------
    train_acc = [acc * 100 for acc in history.history["accuracy"]]
    val_acc = [acc * 100 for acc in history.history["val_accuracy"]]

    final_train = train_acc[-1]
    final_val = val_acc[-1]

    axes[0].plot(
        epochs, train_acc, 
        label=f"Train Accuracy: {final_train:.2f}%", 
        color="#8A2BE2", marker="o", markersize=4, linewidth=1.8, alpha=0.9
    )
    axes[0].plot(
        epochs, val_acc, 
        label=f"Validation Accuracy: {final_val:.2f}%", 
        color="#2E86AB", marker="d", markersize=4, linewidth=1.8, alpha=0.9
    )

    axes[0].set_title("Training Performance (Accuracy)", fontsize=13, fontweight="bold", pad=12)
    axes[0].set_xlabel("Epochs", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    axes[0].grid(True, linestyle="--", alpha=0.6, color="#B0B0B0")
    axes[0].legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#D3D3D3")

    # ---------------- กราฟ Loss ----------------
    axes[1].plot(
        epochs, history.history["loss"], 
        label=f"Train Loss: {history.history['loss'][-1]:.4f}", 
        color="#D9534F", marker="s", markersize=4, linewidth=1.8, alpha=0.9
    )
    axes[1].plot(
        epochs, history.history["val_loss"], 
        label=f"Validation Loss: {history.history['val_loss'][-1]:.4f}", 
        color="#5CB85C", marker="^", markersize=4, linewidth=1.8, alpha=0.9
    )

    axes[1].set_title("Training Loss", fontsize=13, fontweight="bold", pad=12)
    axes[1].set_xlabel("Epochs", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Loss", fontsize=11, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.6, color="#B0B0B0")
    axes[1].legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#D3D3D3")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {save_path}")