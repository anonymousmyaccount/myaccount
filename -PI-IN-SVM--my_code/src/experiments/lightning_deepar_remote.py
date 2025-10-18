#!/usr/bin/env python3
import os
import sys
import subprocess
import time

def connect_lightning():
    """Connect to Lightning.ai to get the SSH address"""
    ssh_command = 'curl -s "https://lightning.ai/setup/ssh?t=d33a653b-30ae-4550-a2c8-ca696adaa6be&s=01jq9c88cepzxqpked4s4y18js" | bash'
    
    try:
        print("Setting up Lightning.ai SSH connection...")
        result = subprocess.run(ssh_command, shell=True, capture_output=True, text=True)
        
        # Extract the SSH address from the output
        output_lines = result.stdout.strip().split('\n')
        ssh_address = None
        
        for line in output_lines:
            if line.strip().startswith('ssh '):
                ssh_address = line.strip()
                break
        
        if ssh_address:
            print(f"Found SSH address: {ssh_address}")
            return ssh_address
        else:
            print("Could not find SSH address in output.")
            print("Output was:", result.stdout)
            return None
    except Exception as e:
        print(f"Error connecting to Lightning.ai: {e}")
        return None

def create_remote_script():
    """Create a Python script to run on the remote Lightning instance"""
    remote_script = """
import time
import pandas as pd
import numpy as np
import os

# Fix for numpy bool deprecation
np.bool = bool

try:
    from gluonts.mx.trainer import Trainer
    from gluonts.model.deepar import DeepAREstimator
    from gluonts.dataset.common import ListDataset
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.run("pip install mxnet gluonts pandas numpy", shell=True, check=True)
    from gluonts.mx.trainer import Trainer
    from gluonts.model.deepar import DeepAREstimator
    from gluonts.dataset.common import ListDataset

def run_deepar_model(filename="temperatures.csv"):
    # 1) Load data
    df = pd.read_csv(
        filename,
        header=0,
        parse_dates=["Date"],
        dayfirst=True,
    )
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
    print(f"Training time (s): {training_time:.2f}")
    print(f"PICP (90% PI): {picp * 100:.1f}%")
    print(f"MPIW (90% PI): {mpiw:.3f}")

if __name__ == "__main__":
    print("Starting DeepAR model script on Lightning.ai...")
    run_deepar_model()
    print("DeepAR model execution completed!")
"""
    
    with open("remote_deepar.py", "w") as f:
        f.write(remote_script)
    
    print("Created remote script file: remote_deepar.py")
    return "remote_deepar.py"

def main():
    # Get the dataset filename
    dataset_filename = "daily-minimum-temperatures-in-me copy.csv"
    
    # Check if the dataset exists
    if not os.path.isfile(dataset_filename):
        print(f"Error: Dataset file '{dataset_filename}' not found!")
        return 1
    
    # Create the remote script
    remote_script = create_remote_script()
    
    # Connect to Lightning.ai
    ssh_address = connect_lightning()
    if not ssh_address:
        print("Failed to get Lightning.ai SSH address.")
        return 1
    
    # Extract just the SSH login part (remove the 'ssh' command)
    ssh_login = ssh_address.replace('ssh ', '').strip()
    
    # Upload the files to Lightning.ai
    try:
        print(f"Uploading dataset ({dataset_filename}) to Lightning.ai...")
        subprocess.run(
            ["scp", dataset_filename, f"{ssh_login}:temperatures.csv"],
            check=True
        )
        
        print(f"Uploading script ({remote_script}) to Lightning.ai...")
        subprocess.run(
            ["scp", remote_script, f"{ssh_login}:deepar.py"],
            check=True
        )
        
        # Run the script on Lightning.ai
        print("Running DeepAR model on Lightning.ai...")
        subprocess.run(
            ["ssh", ssh_login, "python deepar.py"],
            check=True
        )
        
        print("Successfully completed DeepAR model execution on Lightning.ai!")
        return 0
    
    except subprocess.CalledProcessError as e:
        print(f"Error during execution: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 