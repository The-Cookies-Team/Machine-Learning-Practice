import numpy as np

# Example between-class scatter matrix (B) and within-class scatter matrix (S)
B = np.array([[1, 0.5], [0.5, 2]])  # Replace with your B matrix
S = np.array([[2, 0.1], [0.1, 1]])  # Replace with your S matrix

# Solve the generalized eigenvalue problem
eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(S) @ B)

# Find the eigenvector corresponding to the largest eigenvalue
max_eigenvalue_index = np.argmax(eigenvalues)
w_optimal = eigenvectors[:, max_eigenvalue_index]

# Normalize w to satisfy w.T @ w = 1
w_optimal /= np.linalg.norm(w_optimal)

# Print results
print("Optimal direction vector w:", w_optimal)
print("Maximum eigenvalue (objective):", eigenvalues[max_eigenvalue_index])
