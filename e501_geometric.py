import numpy as np


def main():
    # Data from Table 1
    X = np.array([5, 0, 2, 1, 2])
    Y = np.array([2, 1, 1, 1, 0])

    # Step 1: Create the design matrix X̄ [1s and X]
    n = len(X)
    X_bar = np.column_stack([np.ones(n), X])

    print("Q1 Solutions:")

    print("\n(a) Computing predicted response vector Ŷ:")

    # Calculate projection matrix P
    P = X_bar @ np.linalg.inv(X_bar.T @ X_bar) @ X_bar.T

    # Calculate predicted Y (Ŷ = PY)
    Y_hat = P @ Y

    print("\nProjection matrix P:")
    print(P)
    print("\nPredicted response vector Ŷ:")
    print(f"Ŷ = {Y_hat}")

    print("\n(b) Computing bias and slope:")

    # Calculate β = (X̄ᵀX̄)⁻¹X̄ᵀY
    beta = np.linalg.inv(X_bar.T @ X_bar) @ X_bar.T @ Y

    bias = beta[0]
    slope = beta[1]

    print(f"\nBias (β₀) = {bias:.4f}")
    print(f"Slope (β₁) = {slope:.4f}")
    print(f"\nRegression equation: Y = {bias:.4f} + {slope:.4f}X")

    # Additional calculations to verify our results
    print("\nVerification:")
    print("R² score:", 1 - np.sum((Y - Y_hat) ** 2) / np.sum((Y - np.mean(Y)) ** 2))
    print("Mean squared error:", np.mean((Y - Y_hat) ** 2))


if __name__ == "__main__":
    main()
