#!/usr/bin/env python3
import time
import os
import sys
import subprocess

print("Python version:", sys.version)
print("Current directory:", os.getcwd())

# First install the necessary packages
print("Installing required packages...")
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "pandas", "numpy", "gluonts", "mxnet"])
    print("Packages installed successfully!")
except Exception as e:
    print(f"Error installing packages: {e}")
    sys.exit(1)

# Force Python to look in the user site-packages
import site
sys.path.append(site.getusersitepackages())

import pandas as pd
import numpy as np

# Fix for numpy bool deprecation
np.bool = bool

print("Importing GluonTS...")
try:
    from gluonts.mx.trainer import Trainer
    from gluonts.model.deepar import DeepAREstimator
    from gluonts.dataset.common import ListDataset
    print("GluonTS imported successfully!")
except Exception as e:
    print(f"Error importing GluonTS: {e}")
    print("Python path:", sys.path)
    print("Trying to list installed packages:")
    subprocess.call([sys.executable, "-m", "pip", "list"])
    sys.exit(1)

def run_deepar_model(filename="temperatures.csv"):
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
    df = df.rename(columns={df.columns[1]: "temp"})
    df = df.set_index("Date")

    # 2) Split 70/30
    n = len(df)
    train_size = int(n * 0.7)
    train_series = df.temp.values[:train_size]
    full_series = df.temp.values  # for test dataset
    
    freq = "D"
    prediction_length = n - train_size

    print(f"Split data: {train_size} training samples, {prediction_length} prediction length")

    # 3) Build GluonTS ListDatasets
    train_ds = ListDataset(
        [{"start": df.index[0], "target": train_series}],
        freq=freq,
    )
    test_ds = ListDataset(
        [{"start": df.index[0], "target": full_series}],
        freq=freq,
    )

    # 4) Define & train estimator
    print("Initializing DeepAR estimator...")
    estimator = DeepAREstimator(
        freq=freq,
        prediction_length=prediction_length,
        trainer=Trainer(
            ctx="cpu",
            epochs=20,
            learning_rate=1e-3,
            batch_size=32,
        ),
    )

    print("Starting model training...")
    start_time = time.time()
    predictor = estimator.train(train_ds)
    training_time = time.time() - start_time

    # 5) Make forecasts
    print("Making forecasts...")
    forecasts = list(predictor.predict(test_ds))
    f = forecasts[0]

    # 6) Compute PICP & MPIW for 90% PI (alpha=0.10)
    alpha = 0.10
    lower_q = alpha / 2
    upper_q = 1 - lower_q

    # extract np arrays of quantiles
    lower = np.array(f.quantile(lower_q))
    upper = np.array(f.quantile(upper_q))
    # actuals in the test window
    actual = full_series[train_size:]

    # PICP = fraction of actuals within [lower, upper]
    picp = np.mean((actual >= lower) & (actual <= upper))
    # MPIW = average width of the interval
    mpiw = np.mean(upper - lower)

    # 7) Print results
    print("\n=== RESULTS ===")
    print(f"Training time (s): {training_time:.2f}")
    print(f"PICP (90% PI): {picp * 100:.1f}%")
    print(f"MPIW (90% PI): {mpiw:.3f}")
    
    return True

if __name__ == "__main__":
    print("Starting DeepAR model on Lightning.ai...")
    run_deepar_model()
    print("DeepAR model execution completed!")
