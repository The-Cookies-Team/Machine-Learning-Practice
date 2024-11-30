import numpy as np
import time
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import precision_score, recall_score, f1_score

# Load dữ liệu Iris
iris = load_iris()
X = iris.data  # Sử dụng tất cả các features (4 đặc trưng)
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


def evaluate_model(model, X_test, y_test, model_name):
    start_time = time.time()
    y_pred = model.predict(X_test)
    end_time = time.time()
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")
    training_time = end_time - start_time

    print(f"\n--- {model_name} ---")
    print(f"Thời gian huấn luyện: {training_time:.4f} giây")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"F1-score (macro): {f1:.4f}")
    print(
        f"Classification Report:\n{classification_report(y_test, y_pred, target_names=target_names)}"
    )
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")


# Huấn luyện và đánh giá LDA (Sử dụng tất cả 4 đặc trưng)
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_scaled, y_train)
evaluate_model(lda, X_test_scaled, y_test, "LDA")


# Huấn luyện và đánh giá SVM (Linear Kernel) (Sử dụng tất cả 4 đặc trưng)
svm_linear = SVC(kernel="linear", C=1)
svm_linear.fit(X_train_scaled, y_train)
evaluate_model(svm_linear, X_test_scaled, y_test, "SVM (Linear Kernel)")


# Huấn luyện và đánh giá SVM (RBF Kernel) (Sử dụng tất cả 4 đặc trưng)
svm_rbf = SVC(kernel="rbf", C=1, gamma=0.1)
svm_rbf.fit(X_train_scaled, y_train)
evaluate_model(svm_rbf, X_test_scaled, y_test, "SVM (RBF Kernel)")


# Vẽ đồ thị Decision Boundary (chỉ 2 features đầu tiên) - để đơn giản, ta chỉ vẽ cho tập huấn luyện
def plot_decision_boundary(X, y, clf, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    # Sử dụng tất cả các đặc trưng khi huấn luyện, nhưng chỉ 2 đặc trưng khi dự đoán
    Z = clf.predict(
        np.c_[
            xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel()), np.zeros_like(xx.ravel())
        ]
    )  # Cung cấp 4 đặc trưng cho mô hình dự đoán
    Z = Z.reshape(xx.shape)

    cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])
    cmap_bold = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

    plt.figure(figsize=(12, 10))
    plt.subplot(1, 2, 1)
    plt.contourf(xx, yy, Z, cmap=cmap_light)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor="k")
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title(title)

    # Confusion Matrix
    plt.subplot(1, 2, 2)
    cm = confusion_matrix(y_test, clf.predict(X_test_scaled))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    plt.tight_layout()
    plt.show()


# Vẽ decision boundary chỉ với 2 đặc trưng đầu tiên (Để dễ hiển thị trên đồ thị)
plot_decision_boundary(X_train_scaled[:, :2], y_train, lda, "LDA Decision Boundary")
plot_decision_boundary(
    X_train_scaled[:, :2], y_train, svm_linear, "SVM (Linear Kernel) Decision Boundary"
)
plot_decision_boundary(
    X_train_scaled[:, :2], y_train, svm_rbf, "SVM (RBF Kernel) Decision Boundary"
)
