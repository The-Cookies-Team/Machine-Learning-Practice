import numpy as np
import matplotlib.pyplot as plt

# Input data
class1 = np.array([[2, 3], [3, 3], [3, 4], [5, 8], [7, 7]])  # Triangles
class2 = np.array([[5, 4], [6, 5], [7, 4], [7, 5], [9, 4], [8, 2]])  # Circles

mu_class1 = np.mean(class1, axis=0)
mu_class2 = np.mean(class2, axis=0)

S_class1 = sum(
    (x - mu_class1).reshape(-1, 1) @ (x - mu_class1).reshape(1, -1) for x in class1
)
S_class2 = sum(
    (x - mu_class2).reshape(-1, 1) @ (x - mu_class2).reshape(1, -1) for x in class2
)
S = S_class1 + S_class2

mean_diff = mu_class1 - mu_class2

epsilon = 1e-5
S_regularized = S + epsilon * np.eye(S.shape[0])

w = np.linalg.solve(S_regularized, mu_class1 - mu_class2)
w0 = -0.5 * (w.T @ (mu_class1 + mu_class2))

plt.figure(figsize=(8, 8))
plt.scatter(class1[:, 0], class1[:, 1], color="blue", marker="^", label="Class 1")
plt.scatter(class2[:, 0], class2[:, 1], color="red", marker="o", label="Class 2")

x_values = np.linspace(0, 10, 100)
y_values = -(w[0] * x_values + w0) / w[1]
plt.plot(x_values, y_values, color="green", label="Linear Discriminant Line")

plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Linear Discriminant Analysis")
plt.legend()
plt.grid()
plt.show()
