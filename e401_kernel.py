import numpy as np

# Input data points from Table 1
data_points = np.array([[4, 2.9], [2.5, 1], [3.5, 4], [2, 2.1]])

# Number of data points
n = data_points.shape[0]

# Initialize kernel matrix K
K = np.zeros((n, n))

# Compute the kernel matrix K using K(x_i, x_j) = ||x_i - x_j||^2
for i in range(n):
    for j in range(n):
        K[i, j] = np.linalg.norm(data_points[i] - data_points[j]) ** 2

# Print the kernel matrix in a clear format
print("Kernel Matrix K:")
print(K)
