import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

# Load dữ liệu Iris và chọn 2 features đầu tiên để vẽ decision boundary
iris = load_iris()
X = iris.data[:, :2]
y = iris.target
feature_names = iris.feature_names[:2]

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Huấn luyện các mô hình
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_scaled, y_train)

svm_linear = SVC(kernel="linear", C=1)  # C=1
svm_linear.fit(X_train_scaled, y_train)

svm_rbf = SVC(kernel="rbf", C=10, gamma=0.1)  # C=1, gamma=0.1
svm_rbf.fit(X_train_scaled, y_train)


# Hàm vẽ decision boundary
def plot_decision_boundary(X, y, clf, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])
    cmap_bold = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_light)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor="k")
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title(title)
    plt.show()


# Vẽ decision boundary cho các mô hình
plot_decision_boundary(X_train_scaled, y_train, lda, "LDA Decision Boundary")
plot_decision_boundary(
    X_train_scaled, y_train, svm_linear, "SVM (Linear Kernel) Decision Boundary"
)
plot_decision_boundary(
    X_train_scaled, y_train, svm_rbf, "SVM (RBF Kernel) Decision Boundary"
)
