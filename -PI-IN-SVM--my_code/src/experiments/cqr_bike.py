# cqr_bike.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from cqr import train_two_model_cqr_conformal


def load_bike_data(file_path='bike_train.csv'):
    # Read the bike sharing dataset
    df = pd.read_csv(file_path)
    # Parse datetime features
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year.map({2011:0, 2012:1})
    # Drop unneeded columns
    df = df.drop(columns=['datetime', 'casual', 'registered'])
    # Separate features and target
    X = df.drop(columns=['count']).values.astype(np.float32)
    y = df['count'].values.astype(np.float32)
    return X, y


if __name__ == '__main__':
    # Load and scale data
    X, y = load_bike_data()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Conformal quantile parameters (using defaults: q_low=0.1, q_high=0.9, alpha=0.1)
    q_low, q_high, alpha = 0.025, 0.975, 0.1
    verbose = False

    coverages, mpiws, training_times = [], [], []
    print("Running CQR model on Bike dataset 10 times...")

    # Perform 10 runs with different random seeds
    for seed in range(10):
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_scaled, y, test_size=0.20, random_state=42 + seed
        )
        X_train, X_cal, y_train, y_cal = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42 + seed
        )

        # Reshape targets for TensorFlow
        y_train = y_train.reshape(-1, 1)
        y_cal = y_cal.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        # Run conformal quantile regression
        t, picp, mw, low_cov, up_cov, model_lo, model_hi, lb, ub = \
            train_two_model_cqr_conformal(
                X_train, y_train, X_cal, y_cal, X_test, y_test,
                q_low=q_low, q_high=q_high, alpha=alpha, verbose=verbose
            )

        training_times.append(t)
        coverages.append(picp)
        mpiws.append(mw)

    # Compute summary statistics
    mean_cov = np.mean(coverages)
    std_cov = np.std(coverages)
    mean_mpiw = np.mean(mpiws)
    std_mpiw = np.std(mpiws)
    mean_time = np.mean(training_times)

    print("\n=== Summary over 10 runs ===")
    print(f"Mean PICP: {mean_cov:.4f} ± {std_cov:.4f}")
    print(f"Mean MPIW: {mean_mpiw:.4f} ± {std_mpiw:.4f}")
    print(f"Mean Training Time: {mean_time:.2f} seconds")

    # Plot the prediction interval for the last run
    y_test_flat = y_test.flatten()
    idx = np.argsort(y_test_flat)
    sorted_y = y_test_flat[idx]
    sorted_lb = lb[idx]
    sorted_ub = ub[idx]
    x = np.arange(len(sorted_y))

    plt.figure(figsize=(10, 6))
    plt.scatter(x, sorted_y, color='blue', alpha=0.6, label='Actual Count')
    plt.fill_between(
        x, sorted_lb, sorted_ub, color='gray', alpha=0.3,
        label=f'Prediction Interval ({1-alpha:.0%})'
    )
    plt.title(
        f'Bike Sharing Count Conformal CQR\n'
        f'Mean PICP: {mean_cov:.4f} ± {std_cov:.4f}, Mean MPIW: {mean_mpiw:.4f} ± {std_mpiw:.4f}'
    )
    plt.xlabel('Sample index (sorted)')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.show() 