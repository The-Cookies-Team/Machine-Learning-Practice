import numpy as np
import matplotlib.pyplot as plt

# Tọa độ các điểm của hai lớp (tam giác và tròn)
class1 = np.array([[2, 3], [3, 3], [3, 4], [5, 8], [7, 7]])  # Ước lượng từ hình ảnh
class2 = np.array([[5, 4], [6, 5], [7, 4], [7, 5], [9, 4], [8, 2]])

# Tính trung bình mỗi lớp
mu_class1 = np.mean(class1, axis=0)
mu_class2 = np.mean(class2, axis=0)

# Tính ma trận tán xạ trong lớp (within-class scatter matrix)
S_class1 = sum(
    (x - mu_class1).reshape(-1, 1) @ (x - mu_class1).reshape(1, -1) for x in class1
)
S_class2 = sum(
    (x - mu_class2).reshape(-1, 1) @ (x - mu_class2).reshape(1, -1) for x in class2
)
S = S_class1 + S_class2

# Hiệu trung bình giữa hai lớp
mean_diff = mu_class1 - mu_class2

# Thêm regularization để tránh ma trận không khả nghịch
epsilon = 1e-5  # Giá trị nhỏ để thêm vào đường chéo của ma trận
S_regularized = S + epsilon * np.eye(S.shape[0])  # Ma trận regularized

# Tính vector w với ma trận regularized
w = np.linalg.solve(S_regularized, mean_diff)

# Vẽ đồ thị
plt.figure(figsize=(8, 8))
plt.scatter(class1[:, 0], class1[:, 1], color="blue", marker="^", label="Class 1")
plt.scatter(class2[:, 0], class2[:, 1], color="red", marker="o", label="Class 2")

# Đường phân tách
x_values = np.linspace(0, 10, 100)
y_values = -(w[0] / w[1]) * x_values  # Đường thẳng: w₁x + w₂y = 0
plt.plot(x_values, y_values, color="green", label="Linear Discriminant Line")

plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Linear Discriminant Analysis")
plt.legend()
plt.grid()
plt.show()
