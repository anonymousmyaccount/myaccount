import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define build_model at module level so it can be used anywhere
def build_model(input_dim):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(200, input_dim=input_dim, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

def train_two_model_cqr_conformal(X_train, y_train, X_cal, y_cal, X_test, y_test, q_low=0.1, q_high=0.9, alpha=0.1, verbose=False):

    # 1.
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)
    y_cal = y_cal.reshape(-1)

    def quantile_loss(q):
        def loss(y_true, y_pred):
            e = y_true - y_pred
            return tf.reduce_mean(tf.maximum(q * e, (q - 1) * e))
        return loss

    # Separate optimizers for each model
    opt_lower = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)  # Using legacy optimizer for M1/M2 Macs
    opt_upper = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)  # Using legacy optimizer for M1/M2 Macs

    model_lo = build_model(X_train.shape[1])
    model_hi = build_model(X_train.shape[1])

    model_lo.compile(loss=quantile_loss(q_low), optimizer=opt_lower)
    model_hi.compile(loss=quantile_loss(q_high), optimizer=opt_upper)

    # 2. Train both models
    start_time = time.time()
    model_lo.fit(X_train, y_train, epochs=400, batch_size=40, verbose=0)
    model_hi.fit(X_train, y_train, epochs=400, batch_size=40, verbose=0)
    training_time = time.time() - start_time

    # 3. Predict quantiles on calibration and test sets
    def predict_quantiles(model, X):
        return model.predict(X, verbose=0).reshape(-1)

    q_lo_cal = predict_quantiles(model_lo, X_cal)
    q_hi_cal = predict_quantiles(model_hi, X_cal)
    q_lo_test = predict_quantiles(model_lo, X_test)
    q_hi_test = predict_quantiles(model_hi, X_test)

    # 4. Conformal adjustment
    scores = np.maximum(q_lo_cal - y_cal, y_cal - q_hi_cal)
    q_hat = np.quantile(scores, 1 - alpha, method='higher')

    lower_bound = q_lo_test - q_hat
    upper_bound = q_hi_test + q_hat

    low_cov = np.sum(y_test <= lower_bound) / len(y_test)
    up_cov = np.sum(upper_bound >= y_test) / len(y_test)
    
    # 5. Evaluation metrics
    coverage = np.mean((y_test >= lower_bound) & (y_test <= upper_bound))
    mpiw = np.mean(upper_bound - lower_bound)

    if verbose:
        print(f"Run {run_counter+1}/10:")
        print(f"Low Coverage: {low_cov:.4f}")
        print(f"High Coverage: {up_cov:.4f}")
        print(f"PICP (conformal): {coverage:.4f}")
        print(f"MPIW (conformal): {mpiw:.4f}")
        print(f"Training time: {training_time:.2f} seconds\n")

    # Return models and bounds for visualization
    return training_time, coverage, mpiw, low_cov, up_cov, model_lo, model_hi, lower_bound, upper_bound

def load_star_data(base_path='.'):
    # load STAR.csv from the provided base_path (defaults to '.')
    df = pd.read_csv(f"{base_path}/STAR.csv")

    # --- Categorical → numeric mappings ---
    df.loc[df['gender']=='female','gender'] = 0
    df.loc[df['gender']=='male',  'gender'] = 1
    # ... all your existing mappings ...

    # Drop any incomplete rows
    df = df.dropna()

    # Build target = sum of reading & math across K–3
    grade = (
        df["readk"] + df["read1"] + df["read2"] + df["read3"]
      + df["mathk"] + df["math1"] + df["math2"] + df["math3"]
    )

    # Remove the uninformative ID and school/system ID columns
    df = df.drop(columns=df.columns[0])  # drop the unnamed student ID
    df = df.drop(columns=[
        'systemk','system1','system2','system3',
        'schoolidk','schoolid1','schoolid2','schoolid3'
    ])
    # Now features = everything except the eight read/math scores
    drop_cols = ['readk','read1','read2','read3',
                 'mathk','math1','math2','math3']
    X = df.drop(columns=drop_cols).values.astype(np.float32)
    y = grade.values.astype(np.float32)

    return X, y

# Main script to run on NO2.txt
if __name__ == "__main__":
    # Load the data
    data = np.loadtxt('NO2.txt', delimiter='\t')
    
    # Assuming first column is the target and the rest are features
    X = data[:, 1:]  # Features (columns 1 to end)
    y = data[:, 0]   # Target (column 0)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Set parameters
    q_low = 0.025
    q_high = 0.975
    alpha = 0.05
    verbose = True
    
    # Lists to store results
    coverages = []
    mpiws = []
    training_times = []
    low_covs = []
    up_covs = []
    
    # Run 10 times with different random seeds
    print("Running CQR model on NO2 dataset 10 times...")
    
    global run_counter
    run_counter = 0
    
    for i in range(10):
        run_counter = i
        # Split the data into train, calibration and test sets (60%, 20%, 20%)
        # Use different random seed each time
        X_temp, X_test, y_temp, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42+i)
        X_train, X_cal, y_train, y_cal = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42+i)
        
        # Reshape data for the model
        y_train = y_train.reshape(-1, 1)
        y_cal = y_cal.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        
        # Run the CQR model
        training_time, coverage, mpiw, low_cov, up_cov, model_lo, model_hi, lower_bound, upper_bound = train_two_model_cqr_conformal(
            X_train, y_train, X_cal, y_cal, X_test, y_test, 
            q_low=q_low, q_high=q_high, alpha=alpha, verbose=verbose
        )
        
        # Store results
        coverages.append(coverage)
        mpiws.append(mpiw)
        training_times.append(training_time)
        low_covs.append(low_cov)
        up_covs.append(up_cov)
    
    # Calculate statistics
    mean_coverage = np.mean(coverages)
    std_coverage = np.std(coverages)
    mean_mpiw = np.mean(mpiws)
    std_mpiw = np.std(mpiws)
    mean_training_time = np.mean(training_times)
    mean_low_cov = np.mean(low_covs)
    mean_up_cov = np.mean(up_covs)
    
    # Print overall results
    print("\n=== Summary Statistics over 10 Runs ===")
    print(f"Mean PICP: {mean_coverage:.4f} ± {std_coverage:.4f}")
    print(f"Mean MPIW: {mean_mpiw:.4f} ± {std_mpiw:.4f}")
    print(f"Mean Lower Bound Coverage: {mean_low_cov:.4f}")
    print(f"Mean Upper Bound Coverage: {mean_up_cov:.4f}")
    print(f"Mean Training Time: {mean_training_time:.2f} seconds")
    
    # Plot results with prediction intervals for the last run
    plt.figure(figsize=(10, 6))
    
    # Ensure all arrays are 1D for plotting
    y_test_1d = y_test.flatten()
    
    # Sort test data for better visualization
    sorted_indices = np.argsort(y_test_1d)
    sorted_y_test = y_test_1d[sorted_indices]
    sorted_lower_bound = lower_bound[sorted_indices]
    sorted_upper_bound = upper_bound[sorted_indices]
    
    # Plot test points and prediction intervals
    x_values = np.arange(len(sorted_y_test))
    plt.scatter(x_values, sorted_y_test, color='blue', label='Actual Values', alpha=0.6)
    plt.fill_between(x_values, sorted_lower_bound, sorted_upper_bound, 
                     color='gray', alpha=0.3, label=f'Prediction Interval ({1-alpha:.0%})')
    
    plt.title(f'NO2 Dataset with Conformal Prediction Intervals\nMean PICP: {mean_coverage:.4f} ± {std_coverage:.4f}, Mean MPIW: {mean_mpiw:.4f} ± {std_mpiw:.4f}')
    plt.xlabel('Sample Index (Sorted by Target Value)')
    plt.ylabel('Target Value')
    plt.legend()
    plt.tight_layout()
    plt.show()