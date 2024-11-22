import numpy as np


def compute_means(data, labels):
    """
    Tính vector trung bình cho mỗi lớp.
    """
    unique_labels = np.unique(labels)
    means = {}
    for label in unique_labels:
        class_data = data[labels == label]
        means[label] = np.mean(class_data, axis=0)
    return means


def compute_between_class_scatter(mu_plus, mu_minus):
    """
    Tính ma trận phân tán giữa các lớp B.
    """
    diff = (mu_plus - mu_minus).reshape(-1, 1)
    return diff @ diff.T


def compute_within_class_scatter(data, labels, means):
    """
    Tính ma trận phân tán trong lớp S.
    """
    S_plus = np.zeros((2, 2))
    S_minus = np.zeros((2, 2))

    # Tính S₊₁
    class_plus = data[labels == 1]
    mu_plus = means[1]
    for x in class_plus:
        diff = (x - mu_plus).reshape(-1, 1)
        S_plus += diff @ diff.T

    # Tính S₋₁
    class_minus = data[labels == -1]
    mu_minus = means[-1]
    for x in class_minus:
        diff = (x - mu_minus).reshape(-1, 1)
        S_minus += diff @ diff.T

    return S_plus + S_minus


def find_best_direction(S, mean_diff):
    """
    Tìm vector hướng w tốt nhất và chuẩn hóa nó.
    """
    epsilon = 1e-5  # Thêm regularization để tránh ma trận S kỳ dị
    S_reg = S + epsilon * np.eye(S.shape[0])  # Regular hóa ma trận S
    w = np.linalg.solve(S_reg, mean_diff)  # Giải hệ phương trình S * w = mean_diff
    # Chuẩn hóa vector w để có độ dài chuẩn
    w = w / np.linalg.norm(w)
    return w


def find_separation_point(w, mu_plus, mu_minus):
    """
    Tìm điểm phân tách tốt nhất trên w.
    """
    # Tính điểm phân tách bằng công thức trung bình trọng số
    return -0.5 * w.T @ (mu_plus + mu_minus)


# Input data
data = np.array([[4.0, 2.9], [3.5, 4.0], [2.5, 1.0], [2.0, 2.1]])
labels = np.array([1, 1, -1, -1])

# (a) Tính μ₊₁, μ₋₁ và ma trận B
means = compute_means(data, labels)
mu_plus = means[1]
mu_minus = means[-1]
B = compute_between_class_scatter(mu_plus, mu_minus)

# (b) Tính ma trận S
S = compute_within_class_scatter(data, labels, means)

# (c) Tìm vector hướng w
mean_diff = mu_plus - mu_minus
w = find_best_direction(S, mean_diff)

# (d) Tìm điểm phân tách
w0 = find_separation_point(w, mu_plus, mu_minus)

# In kết quả
print(f"μ₊₁ = {mu_plus}")
print(f"μ₋₁ = {mu_minus}")
print("\nThe between-class scatter matrix B:\n", B)
print("\nThe within-class scatter matrix S:\n", S)
print("\nThe best direction vector w:\n", w)
print(f"\nThe separation point w₀: {w0}")


# Kiểm tra phân loại
def classify(x, w, w0):
    return np.sign(w.T @ x + w0)


# Kiểm tra kết quả phân loại trên tập dữ liệu
for i, x in enumerate(data):
    prediction = classify(x, w, w0)
    print(f"\nData {i+1}: {x}")
    print(f"Label: {labels[i]}")
    print(f"Predict: {prediction}")
