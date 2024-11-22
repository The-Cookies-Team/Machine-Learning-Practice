import numpy as np
from scipy import linalg


def gaussian_kernel(x1, x2, sigma_squared=5):
    return np.exp(-np.sum((x1 - x2) ** 2) / (2 * sigma_squared))


def main():
    # Given points
    x1 = np.array([2.5, 1])
    x2 = np.array([3.5, 4])
    x3 = np.array([2, 2.1])
    points = [x1, x2, x3]

    print("Q4 Solutions:")
    print("\n(a) Computing the Gaussian kernel matrix (σ²=5):")

    # Calculate kernel matrix
    n = len(points)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = gaussian_kernel(points[i], points[j])

    print("\nKernel matrix K:")
    print(K)

    print("\n(b) Computing distance of φ(x1) from the mean in feature space:")

    # Calculate distance using kernel trick
    # ||φ(x1) - μ||² = k(x1,x1) - 2/n Σk(x1,xi) + 1/n² ΣΣk(xi,xj)
    k11 = K[0, 0]
    sum_k1i = np.sum(K[0, :])
    sum_kij = np.sum(K)

    distance_squared = k11 - (2 / n) * sum_k1i + (1 / n**2) * sum_kij
    distance = np.sqrt(abs(distance_squared))  # abs to handle numerical errors

    print(f"\nDistance = {distance:.6f}")

    print("\n(c) Computing dominant eigenvector and eigenvalue:")

    # Compute eigenvalues and eigenvectors
    eigenvals, eigenvecs = linalg.eigh(K)

    # Get dominant (largest) eigenvalue and corresponding eigenvector
    dominant_eigenval = eigenvals[-1]
    dominant_eigenvec = eigenvecs[:, -1]

    print("\nDominant eigenvalue:")
    print(f"{dominant_eigenval:.6f}")

    print("\nDominant eigenvector:")
    print(dominant_eigenvec)


if __name__ == "__main__":
    main()
