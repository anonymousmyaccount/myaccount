import time
import pandas as pd
import numpy as np
import subprocess
import os
import sys
import shutil

# Fix for numpy bool deprecation
np.bool = bool

def connect_to_lightning():
    """Connect to Lightning.ai using the provided SSH command."""
    ssh_command = 'curl -s "https://lightning.ai/setup/ssh?t=d33a653b-30ae-4550-a2c8-ca696adaa6be&s=01jq9c88cepzxqpked4s4y18js" | bash'
    try:
        print("Connecting to Lightning.ai...")
        process = subprocess.run(ssh_command, shell=True, check=True, text=True)
        print("Successfully connected to Lightning.ai")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error connecting to Lightning.ai: {e}")
        return False

def upload_dataset(filename="daily-minimum-temperatures-in-me copy.csv"):
    """Upload the dataset to Lightning.ai if needed."""
    print(f"Checking if dataset '{filename}' needs to be uploaded...")
    
    # First, confirm the file exists locally
    if not os.path.isfile(filename):
        print(f"Error: Dataset file '{filename}' not found in the current directory.")
        return False
    
    # Check if we're already on Lightning.ai
    # If a ~/.lightning directory exists, we're likely already on Lightning
    if os.path.isdir(os.path.expanduser("~/.lightning")):
        # We're on Lightning, check if the file is already there
        if os.path.isfile(os.path.join(os.path.expanduser("~/"), filename)):
            print(f"Dataset already available on Lightning.ai.")
            return True
        else:
            # Copy the file to the home directory of the Lightning instance
            try:
                # Assuming we have a different working directory, copy to home
                shutil.copy(filename, os.path.expanduser("~/"))
                print(f"Dataset copied to Lightning.ai home directory.")
                return True
            except Exception as e:
                print(f"Error copying dataset to Lightning.ai: {e}")
                return False
    else:
        # We're not on Lightning yet, the file will be uploaded when we connect
        print("Not yet connected to Lightning.ai. Dataset will be uploaded upon connection.")
        return True

def setup_environment():
    """Install required packages on Lightning.ai instance."""
    print("Setting up environment on Lightning.ai...")
    try:
        subprocess.run("pip install mxnet gluonts pandas numpy", shell=True, check=True)
        print("Environment setup complete.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error setting up environment: {e}")
        return False

def run_deepar_model(filename="daily-minimum-temperatures-in-me copy.csv"):
    """Run the DeepAR model using GluonTS."""
    try:
        from gluonts.mx.trainer import Trainer
        from gluonts.model.deepar import DeepAREstimator
        from gluonts.dataset.common import ListDataset
    except ImportError:
        print("Failed to import GluonTS modules. Make sure environment is properly set up.")
        return False

    # 1) Load data
    try:
        df = pd.read_csv(
            filename,
            header=0,
            parse_dates=["Date"],
            dayfirst=True,
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False
        
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

    # 3) Build GluonTS ListDatasets
    train_ds = ListDataset(
        [{"start": df.index[0], "target": train_series}],
        freq=freq,
    )
    test_ds = ListDataset(
        [{"start": df.index[0], "target": full_series}],
        freq=freq,
        # GluonTS will only forecast the last `prediction_length` points
    )

    # 4) Define & train estimator
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

    start_time = time.time()
    predictor = estimator.train(train_ds)
    training_time = time.time() - start_time

    # 5) Make forecasts
    forecasts = list(predictor.predict(test_ds))
    # we only have one time series, so take forecasts[0]
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
    print(f"Training time (s): {training_time:.2f}")
    print(f"PICP (90% PI): {picp * 100:.1f}%")
    print(f"MPIW (90% PI): {mpiw:.3f}")
    
    return True

def main():
    # Dataset file name
    dataset_filename = "daily-minimum-temperatures-in-me copy.csv"
    
    # Upload dataset if needed
    if not upload_dataset(dataset_filename):
        print("Failed to handle dataset. Exiting.")
        sys.exit(1)
    
    # Connect to Lightning.ai
    if not connect_to_lightning():
        print("Failed to connect to Lightning.ai. Exiting.")
        sys.exit(1)
    
    # Setup environment
    if not setup_environment():
        print("Failed to set up environment. Exiting.")
        sys.exit(1)
    
    # Run the DeepAR model
    if not run_deepar_model(dataset_filename):
        print("Failed to run DeepAR model. Exiting.")
        sys.exit(1)
    
    print("DeepAR model successfully run on Lightning.ai!")

if __name__ == "__main__":
    main() 