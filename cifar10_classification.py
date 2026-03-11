"""
CIFAR-10 Classical Classification
==================================
Implements and evaluates Logistic Regression (Softmax), Linear SVM, and KNN
on the CIFAR-10 dataset with hyperparameter tuning, confusion matrices,
and model comparison.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# 1. Data Loading & Preprocessing
# ─────────────────────────────────────────────

DATA_DIR = os.path.join(
    os.path.dirname(__file__), "intro_to_ai", "datasets", "cifar-10-batches-py"
)
plot_dir = "plots"
os.makedirs(plot_dir, exist_ok=True)

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def load_batch(filepath: str):
    """Load a single CIFAR-10 batch file."""
    with open(filepath, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    data = batch[b"data"]  # (10000, 3072)
    labels = np.array(batch[b"labels"])
    return data, labels


def load_cifar10():
    """Load all CIFAR-10 training + test data."""
    train_data, train_labels = [], []
    for i in range(1, 6):
        d, l = load_batch(os.path.join(DATA_DIR, f"data_batch_{i}"))
        train_data.append(d)
        train_labels.append(l)
    train_data = np.concatenate(train_data)  # (50000, 3072)
    train_labels = np.concatenate(train_labels)  # (50000,)

    test_data, test_labels = load_batch(os.path.join(DATA_DIR, "test_batch"))
    return train_data, train_labels, test_data, test_labels


print("Loading CIFAR-10 dataset...")
X_full, y_full, X_test, y_test = load_cifar10()

# Normalize pixels to [0, 1]
X_full = X_full.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

# Train / Validation split (80% / 20%)
X_train, X_val, y_train, y_val = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
)

# Subsample data to speed up execution considerably
NUM_TRAIN = 3000
NUM_VAL = 1000
X_train = X_train[:NUM_TRAIN]
y_train = y_train[:NUM_TRAIN]
X_val = X_val[:NUM_VAL]
y_val = y_val[:NUM_VAL]

# Standardize to zero mean and unit variance for fast convergence of linear models
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

print(f"Training set:   {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set:       {X_test.shape[0]} samples")
print(f"Feature dim:    {X_train.shape[1]}")
print()

# ─────────────────────────────────────────────
# Helper: plot style
# ─────────────────────────────────────────────
plt.rcParams.update(
    {
        "figure.figsize": (8, 6),
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)

# ─────────────────────────────────────────────
# 2. Logistic Regression (Softmax)
# ─────────────────────────────────────────────
print("=" * 60)
print("LOGISTIC REGRESSION (Softmax)")
print("=" * 60)

lr_C_values = [0.00001, 0.0001, 0.001, 0.01, 0.1]
lr_val_accs = []

for C in lr_C_values:
    print(f"  Training with C={C} ...", end=" ", flush=True)
    model = LogisticRegression(
        C=C,
        solver="lbfgs",
        max_iter=100,
        random_state=42,
    )
    model.fit(X_train, y_train)
    acc = accuracy_score(y_val, model.predict(X_val))
    lr_val_accs.append(acc)
    print(f"val accuracy = {acc:.4f}")

best_lr_idx = int(np.argmax(lr_val_accs))
best_lr_C = lr_C_values[best_lr_idx]
best_lr_acc = lr_val_accs[best_lr_idx]
print(f"\n  Best C = {best_lr_C}  |  Val accuracy = {best_lr_acc:.4f}\n")

# Hyperparameter plot
fig, ax = plt.subplots()
ax.plot(lr_C_values, lr_val_accs, "o-", color="#2563eb", linewidth=2, markersize=8)
ax.set_xscale("log")
ax.set_xlabel("Regularization parameter C")
ax.set_ylabel("Validation Accuracy")
ax.set_title("Logistic Regression – Validation Accuracy vs C")
ax.grid(True, alpha=0.3)
for i, (c, a) in enumerate(zip(lr_C_values, lr_val_accs)):
    ax.annotate(
        f"{a:.4f}",
        (c, a),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontsize=9,
    )
fig.tight_layout()
fig.savefig(os.path.join(plot_dir, "cifar10_lr_hyperparam.png"), dpi=150)
plt.close(fig)
print("  Saved: cifar10_lr_hyperparam.png")

# Retrain best and confusion matrix
lr_best = LogisticRegression(
    C=best_lr_C,
    solver="lbfgs",
    max_iter=100,
    random_state=42,
)
lr_best.fit(X_train, y_train)
lr_val_pred = lr_best.predict(X_val)

fig, ax = plt.subplots(figsize=(10, 8))
cm = confusion_matrix(y_val, lr_val_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
disp.plot(ax=ax, cmap="Blues", values_format="d", xticks_rotation=45)
ax.set_title(f"Logistic Regression Confusion Matrix (Val Acc = {best_lr_acc:.4f})")
fig.tight_layout()
fig.savefig(os.path.join(plot_dir, "cifar10_lr_confusion.png"), dpi=150)
plt.close(fig)
print("  Saved: cifar10_lr_confusion.png\n")


# ─────────────────────────────────────────────
# 3. Linear SVM
# ─────────────────────────────────────────────
print("=" * 60)
print("LINEAR SVM")
print("=" * 60)

svm_C_values = [0.000001, 0.00001, 0.0001, 0.001, 0.01]
svm_val_accs = []

for C in svm_C_values:
    print(f"  Training with C={C} ...", end=" ", flush=True)
    model = LinearSVC(C=C, max_iter=100, random_state=42, dual="auto")
    model.fit(X_train, y_train)
    acc = accuracy_score(y_val, model.predict(X_val))
    svm_val_accs.append(acc)
    print(f"val accuracy = {acc:.4f}")

best_svm_idx = int(np.argmax(svm_val_accs))
best_svm_C = svm_C_values[best_svm_idx]
best_svm_acc = svm_val_accs[best_svm_idx]
print(f"\n  Best C = {best_svm_C}  |  Val accuracy = {best_svm_acc:.4f}\n")

# Hyperparameter plot
fig, ax = plt.subplots()
ax.plot(svm_C_values, svm_val_accs, "o-", color="#dc2626", linewidth=2, markersize=8)
ax.set_xscale("log")
ax.set_xlabel("Regularization parameter C")
ax.set_ylabel("Validation Accuracy")
ax.set_title("Linear SVM – Validation Accuracy vs C")
ax.grid(True, alpha=0.3)
for i, (c, a) in enumerate(zip(svm_C_values, svm_val_accs)):
    ax.annotate(
        f"{a:.4f}",
        (c, a),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontsize=9,
    )
fig.tight_layout()
fig.savefig(os.path.join(plot_dir, "cifar10_svm_hyperparam.png"), dpi=150)
plt.close(fig)
print("  Saved: cifar10_svm_hyperparam.png")

# Retrain best and confusion matrix
svm_best = LinearSVC(C=best_svm_C, max_iter=100, random_state=42, dual="auto")
svm_best.fit(X_train, y_train)
svm_val_pred = svm_best.predict(X_val)

fig, ax = plt.subplots(figsize=(10, 8))
cm = confusion_matrix(y_val, svm_val_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
disp.plot(ax=ax, cmap="Reds", values_format="d", xticks_rotation=45)
ax.set_title(f"Linear SVM Confusion Matrix (Val Acc = {best_svm_acc:.4f})")
fig.tight_layout()
fig.savefig(os.path.join(plot_dir, "cifar10_svm_confusion.png"), dpi=150)
plt.close(fig)
print("  Saved: cifar10_svm_confusion.png\n")


# ─────────────────────────────────────────────
# 4. K-Nearest Neighbors (KNN)
# ─────────────────────────────────────────────
print("=" * 60)
print("K-NEAREST NEIGHBORS")
print("=" * 60)

knn_k_values = range(1, 50, 2)
knn_val_accs = []

for k in knn_k_values:
    print(f"  Training with k={k} ...", end=" ", flush=True)
    model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_val, model.predict(X_val))
    knn_val_accs.append(acc)
    print(f"val accuracy = {acc:.4f}")

best_knn_idx = int(np.argmax(knn_val_accs))
best_knn_k = knn_k_values[best_knn_idx]
best_knn_acc = knn_val_accs[best_knn_idx]
print(f"\n  Best k = {best_knn_k}  |  Val accuracy = {best_knn_acc:.4f}\n")

# Hyperparameter plot
fig, ax = plt.subplots()
ax.plot(knn_k_values, knn_val_accs, "o-", color="#16a34a", linewidth=2, markersize=8)
ax.set_xlabel("Number of neighbors (k)")
ax.set_ylabel("Validation Accuracy")
ax.set_title("KNN – Validation Accuracy vs k")
ax.set_xticks(knn_k_values)
ax.grid(True, alpha=0.3)
for i, (kv, a) in enumerate(zip(knn_k_values, knn_val_accs)):
    ax.annotate(
        f"{a:.4f}",
        (kv, a),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontsize=9,
    )
fig.tight_layout()
fig.savefig(os.path.join(plot_dir, "cifar10_knn_hyperparam.png"), dpi=150)
plt.close(fig)
print("  Saved: cifar10_knn_hyperparam.png")

# Retrain best and confusion matrix
knn_best = KNeighborsClassifier(n_neighbors=best_knn_k, n_jobs=-1)
knn_best.fit(X_train, y_train)
knn_val_pred = knn_best.predict(X_val)

fig, ax = plt.subplots(figsize=(10, 8))
cm = confusion_matrix(y_val, knn_val_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
disp.plot(ax=ax, cmap="Greens", values_format="d", xticks_rotation=45)
ax.set_title(f"KNN Confusion Matrix (Val Acc = {best_knn_acc:.4f})")
fig.tight_layout()
fig.savefig(os.path.join(plot_dir, "cifar10_knn_confusion.png"), dpi=150)
plt.close(fig)
print("  Saved: cifar10_knn_confusion.png\n")


# ─────────────────────────────────────────────
# 5. Model Comparison & Final Test Evaluation
# ─────────────────────────────────────────────
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

results = {
    "Logistic Regression": {
        "acc": best_lr_acc,
        "best_param": f"C={best_lr_C}",
        "model": lr_best,
    },
    "Linear SVM": {
        "acc": best_svm_acc,
        "best_param": f"C={best_svm_C}",
        "model": svm_best,
    },
    "KNN": {"acc": best_knn_acc, "best_param": f"k={best_knn_k}", "model": knn_best},
}

print(f"\n  {'Model':<25} {'Best Hyperparam':<20} {'Val Accuracy':<15}")
print("  " + "-" * 60)
for name, info in results.items():
    print(f"  {name:<25} {info['best_param']:<20} {info['acc']:.4f}")

# Select best model
best_model_name = max(results, key=lambda k: results[k]["acc"])
best_model = results[best_model_name]["model"]
best_val_acc = results[best_model_name]["acc"]

print(f"\n  ★ Best model: {best_model_name} (val accuracy = {best_val_acc:.4f})")

# Final evaluation on test set
test_pred = best_model.predict(X_test)
test_acc = accuracy_score(y_test, test_pred)
print(f"\n  ★ TEST ACCURACY ({best_model_name}): {test_acc:.4f}")
print()

# ─────────────────────────────────────────────
# 6. Discussion / Analysis Summary
# ─────────────────────────────────────────────
print("=" * 60)
print("ANALYSIS SUMMARY")
print("=" * 60)
print("""
1. Model Performance:
   - Logistic Regression and Linear SVM are both linear models and tend to
     achieve similar performance on CIFAR-10 (typically ~37-40% accuracy).
   - KNN can sometimes capture local non-linear patterns but is severely
     impacted by the curse of dimensionality in 3072-dim space.

2. Effect of High Input Dimensionality:
   - CIFAR-10 images have 3072 features (32x32x3). In such high-dimensional
     spaces, distances between points become less meaningful, which hurts
     distance-based methods like KNN.
   - Linear models struggle because the raw pixel representation does not
     provide linearly separable class boundaries.

3. Limitations of Classical Models on Image Data:
   - Classical models operate on raw pixel features and cannot learn
     hierarchical or spatial representations the way deep learning models
     (e.g., CNNs) can.
   - They are invariant to neither translation nor rotation, meaning small
     shifts in the image drastically change the feature vector.
   - Feature engineering (e.g., HOG, PCA) could help, but deep learning
     approaches fundamentally solve these issues by learning features
     end-to-end.
""")

print("All done! Check the generated PNG plots.")
