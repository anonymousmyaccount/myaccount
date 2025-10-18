
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
