import numpy as np
from scipy.stats import norm


# Define the portfolio value function
def portfolio_value(x):
    # Assuming a linear portfolio value function
    return np.dot(x, weights)


# Define the loss function
def loss_function(x):
    portfolio_val = portfolio_value(x)
    return -portfolio_val  # Negative portfolio value for loss function


# Define the target probability level (e.g., 5%)
alpha = 0.05

mean = np.array([0, 0])
cov = np.array([[1, 0], [0, 1]])

# Generate sample paths from the proposal distribution
n_samples = 10000
samples = np.random.multivariate_normal(mean, cov, n_samples)

# Compute the importance weights
weights = norm.pdf(samples) / norm.pdf(samples, mean, cov)

# Compute the portfolio losses
losses = np.array([loss_function(sample) for sample in samples])

# Sort the losses in descending order
sorted_losses = np.sort(losses)[::-1]

# Compute the VaR and CVaR estimates
var_estimate = sorted_losses[int(alpha * n_samples)]
cvar_estimate = np.mean(sorted_losses[:int(alpha * n_samples)])

# Print the VaR and CVaR estimates
print("VaR estimate:", var_estimate)
print("CVaR estimate:", cvar_estimate)
