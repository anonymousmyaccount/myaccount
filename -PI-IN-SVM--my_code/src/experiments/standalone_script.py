#!/usr/bin/env python3
import time
import os
import sys
import traceback
import numpy as np
import pandas as pd

print("Python version:", sys.version)
print("Current directory:", os.getcwd())
print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)

def run_deepar_model(filename="temperatures.csv"):
    try:
        print(f"Loading data from {filename}...")
        # Check if file exists
        if not os.path.exists(filename):
            print(f"Error: File {filename} not found!")
            return False
            
        # 1) Load data
        df = pd.read_csv(
            filename,
            header=0,
            parse_dates=["Date"],
            dayfirst=True,
        )
        print(f"Loaded {len(df)} records from the dataset.")
        
        df = df.sort_values("Date")
        print("Column names:", df.columns)
        df = df.rename(columns={df.columns[1]: "temp"})
        df = df.set_index("Date")

        # 2) Split 70/30
        n = len(df)
        train_size = int(n * 0.7)
        train_series = df.temp.values[:train_size]
        test_series = df.temp.values[train_size:]
        
        print(f"Split data: {train_size} training samples, {len(test_series)} test samples")

        # Since we can't use DeepAR, we'll implement a simple autoregressive model manually
        print("Implementing a simple autoregressive model...")
        
        # Use a simple AR(p) model for forecasting
        p = 7  # Order of AR model (e.g., use last 7 days to predict next day)
        
        # Train a simple AR model
        start_time = time.time()
        
        # We'll compute coefficients manually using least squares
        X = []
        y = []
        
        for i in range(p, len(train_series)):
            X.append(train_series[i-p:i])
            y.append(train_series[i])
        
        X = np.array(X)
        y = np.array(y)
        
        print("X shape:", X.shape, "y shape:", y.shape)
        
        # Compute coefficients using least squares (X'X)^-1 X'y
        coeffs = np.linalg.inv(X.T @ X) @ X.T @ y
        
        training_time = time.time() - start_time
        print("Coefficients:", coeffs)
        
        # Make predictions on test set
        predictions = []
        lower_bounds = []
        upper_bounds = []
        
        # We need the last p values from training to start predictions
        last_p_values = list(train_series[-p:])
        
        # Set a fixed standard deviation for prediction intervals
        # This is estimated from training data errors
        train_preds = X @ coeffs
        residuals = y - train_preds
        std_dev = np.std(residuals)
        
        # For a 90% confidence interval with normal distribution
        z_score = 1.645
        
        for i in range(len(test_series)):
            # Make point prediction
            point_pred = np.sum(np.array(last_p_values[-p:]) * coeffs)
            predictions.append(point_pred)
            
            # Create prediction intervals
            lower = point_pred - z_score * std_dev
            upper = point_pred + z_score * std_dev
            
            lower_bounds.append(lower)
            upper_bounds.append(upper)
            
            # Update window with actual value
            last_p_values.append(test_series[i])
            last_p_values = last_p_values[1:]  # Remove oldest value

        # Calculate PICP - prediction interval coverage probability
        actuals = test_series
        picp = np.mean((actuals >= lower_bounds) & (actuals <= upper_bounds))
        
        # Calculate MPIW - mean prediction interval width
        mpiw = np.mean(np.array(upper_bounds) - np.array(lower_bounds))

        # Print results
        print("\n=== RESULTS ===")
        print(f"Training time (s): {training_time:.2f}")
        print(f"PICP (90% PI): {picp * 100:.1f}%")
        print(f"MPIW (90% PI): {mpiw:.3f}")
        
        return True
    except Exception as e:
        print(f"Error in run_deepar_model: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        print("Starting autoregressive model on Lightning.ai...")
        run_deepar_model()
        print("Model execution completed!")
    except Exception as e:
        print(f"Error in main: {e}")
        traceback.print_exc()
