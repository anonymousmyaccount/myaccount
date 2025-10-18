import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# Create synthetic data
n_samples = 768
n_features = 8

# Create feature matrix with random values
X = np.random.rand(n_samples, n_features)

# Create target variable with a realistic relationship to features
# Y = 0.5*X1 + 0.2*X2 - 0.7*X3 + 0.1*X4 + noise
y = 0.5 * X[:, 0] + 0.2 * X[:, 1] - 0.7 * X[:, 2] + 0.1 * X[:, 3] + 0.3 * np.random.randn(n_samples)

# Create a DataFrame
data = pd.DataFrame(X, columns=[f'X{i+1}' for i in range(n_features)])
data['Y'] = y

# Save to CSV
data.to_csv('energy_efficiency.csv', index=False)

print(f"Created Energy Efficiency dataset with {n_samples} samples and {n_features} features")
print(f"Data shape: {data.shape}")
print(f"First few rows:")
print(data.head()) 