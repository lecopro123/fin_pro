import numpy as np


# Define the transformation function
def transform(X, u, l, rho):
    kappa = np.log(1 + np.abs(X)) / (
        rho * np.linalg.norm(np.log(1 + np.abs(X)), ord=np.inf))
    Zi = 1 * ((u / l)**kappa)
    return Zi


# Generate a random 2x1000 matrix X
X = np.random.rand(2, 1000)

# Define the values for u, l, and rho
u = 2.5
l = 1.3
rho = 0.00001

# Apply the transformation to each value of X
Z = transform(X, u, l, rho)

# Print the transformed matrix Z
print("Transformed matrix Z:")
print(Z)
