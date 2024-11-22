import numpy as np


def calculate_hyperplane(p1, p2):
    """
    Calculate the equation of a hyperplane passing through two points (p1 and p2).
    Args:
        p1, p2 (tuple): Points (x, y) through which the hyperplane passes.
    Returns:
        tuple: Coefficients (a, b, c) of the hyperplane (ax + by + c = 0).
    """
    x1, y1 = p1
    x2, y2 = p2
    a = y2 - y1
    b = x1 - x2
    c = -(a * x1 + b * y1)
    return a, b, c


def calculate_margin(a, b):
    """
    Calculate the margin of a hyperplane.
    Args:
        a, b (float): Coefficients of the hyperplane.
    Returns:
        float: Margin (1 / sqrt(a^2 + b^2)).
    """
    return 1 / np.sqrt(a**2 + b**2)


def find_best_hyperplane(class_1_points, class_2_points):
    """
    Compute the best separating hyperplane for two classes.
    Args:
        class_1_points, class_2_points (numpy array): Points of class 1 and 2.
    Returns:
        tuple: Best hyperplane coefficients, midpoint, centroids.
    """
    centroid_c1 = np.mean(class_1_points, axis=0)
    centroid_c2 = np.mean(class_2_points, axis=0)
    midpoint = (centroid_c1 + centroid_c2) / 2
    direction_vector = centroid_c2 - centroid_c1
    a, b = direction_vector
    c = -(a * midpoint[0] + b * midpoint[1])
    return (a, b, c), midpoint, centroid_c1, centroid_c2


# Input: Dynamic data for classes
# Example data
class_1_points = np.array([[5, 3], [6, 7]])  # Triangles
class_2_points = np.array([[4, 2], [7, 6]])  # Circles

# (a) Find the equations of the two hyperplanes
h1_p1, h1_p2 = (2, 9), (6, 1)  # Points on h1
h2_p1, h2_p2 = (5, 0), (9, 9)  # Points on h2

h1_coeffs = calculate_hyperplane(h1_p1, h1_p2)
h2_coeffs = calculate_hyperplane(h2_p1, h2_p2)

# (b) Identify the support vectors
support_vectors_h1 = {"class_1": [np.array([5, 3])], "class_2": [np.array([4, 2])]}
support_vectors_h2 = {"class_1": [np.array([6, 7])], "class_2": [np.array([7, 6])]}

# (c) Compare margins
margin_h1 = calculate_margin(h1_coeffs[0], h1_coeffs[1])
margin_h2 = calculate_margin(h2_coeffs[0], h2_coeffs[1])

better_hyperplane = "h1" if margin_h1 > margin_h2 else "h2"

# (d) Find the best separating hyperplane
best_hyperplane_coeffs, midpoint, centroid_c1, centroid_c2 = find_best_hyperplane(
    class_1_points, class_2_points
)

# Display results
print("Part (a):")
print(f"Equation of h1: {h1_coeffs[0]}x + {h1_coeffs[1]}y + {h1_coeffs[2]} = 0")
print(f"Equation of h2: {h2_coeffs[0]}x + {h2_coeffs[1]}y + {h2_coeffs[2]} = 0")

print("\nPart (b):")
print(f"Support vectors for h1: {support_vectors_h1}")
print(f"Support vectors for h2: {support_vectors_h2}")

print("\nPart (c):")
print(f"Margin of h1: {margin_h1:.4f}")
print(f"Margin of h2: {margin_h2:.4f}")
print(f"Better hyperplane: {better_hyperplane}")

print("\nPart (d):")
print(
    f"Best separating hyperplane: {best_hyperplane_coeffs[0]:.2f}x + {best_hyperplane_coeffs[1]:.2f}y + {best_hyperplane_coeffs[2]:.2f} = 0"
)
print(f"Midpoint of margin: {midpoint}")
print(f"Centroid of class 1: {centroid_c1}")
print(f"Centroid of class 2: {centroid_c2}")
