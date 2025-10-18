import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler
import time
import warnings
warnings.filterwarnings('ignore')

# For reproducibility
np.random.seed(42)

def load_no2_data():
    """Load NO2 dataset or create synthetic data if not available."""
    try:
        data = np.loadtxt('NO2.txt', delimiter='\t')
        print("Using NO2 dataset from local file")
        
        X = data[:, 1:]  # Features (all columns except the first)
        y = data[:, 0]   # Target (first column)
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        return X, y, feature_names
    except:
        print("NO2 dataset not found. Creating synthetic data.")
        n_samples = 500
        n_features = 8
        X = np.random.randn(n_samples, n_features)
        y = 0.5 * X[:, 0] + 0.2 * X[:, 1] - 0.7 * X[:, 2] + 0.1 * X[:, 3] + 0.3 * np.random.randn(n_samples)
        
        feature_names = [f'feature_{i}' for i in range(n_features)]
        return X, y, feature_names

def svqr_cp(X_train, y_train, X_test, y_test, alpha=0.1, quantile_lower=0.05, quantile_upper=0.95, sparse=False):
    """Implement SVQR+CP or SSVQR+CP method."""
    # Split training data into training and calibration sets
    X_train_proper, X_cal, y_train_proper, y_cal = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42
    )
    
    # Start timer
    start_time = time.time()
    
    # Train SVQR models (using QuantileRegressor)
    # For sparse version, use a slightly higher alpha to encourage sparsity
    alpha_value = 0.01 if sparse else 0.001
    
    # QuantileRegressor in scikit-learn uses L1 penalty by default
    qr_lower = QuantileRegressor(quantile=quantile_lower, alpha=alpha_value, solver='highs')
    qr_upper = QuantileRegressor(quantile=quantile_upper, alpha=alpha_value, solver='highs')
    
    qr_lower.fit(X_train_proper, y_train_proper)
    qr_upper.fit(X_train_proper, y_train_proper)
    
    # Compute nonconformity scores on calibration set
    errors = []
    for i in range(len(X_cal)):
        y_pred_l = qr_lower.predict(X_cal[i].reshape(1, -1))[0]
        y_pred_u = qr_upper.predict(X_cal[i].reshape(1, -1))[0]
        
        # Nonconformity score
        error = max(y_pred_l - y_cal[i], y_cal[i] - y_pred_u, 0)
        errors.append(error)
    
    # Determine calibration quantile
    n_cal = len(errors)
    errors_sorted = np.sort(errors)
    k = int(np.ceil((1 - alpha) * (n_cal + 1)))
    Q = errors_sorted[min(k-1, n_cal-1)]  # Ensure index doesn't exceed array bounds
    
    # Apply conformal prediction on test set
    y_pred_l_test = qr_lower.predict(X_test)
    y_pred_u_test = qr_upper.predict(X_test)
    
    # Adjust with conformal quantile
    y_pred_l_adjusted = y_pred_l_test - Q
    y_pred_u_adjusted = y_pred_u_test + Q
    
    # Compute metrics
    inside = np.logical_and(y_test >= y_pred_l_adjusted, y_test <= y_pred_u_adjusted)
    picp = np.mean(inside)
    mpiw = np.mean(y_pred_u_adjusted - y_pred_l_adjusted)
    
    # End timer
    end_time = time.time()
    
    return {
        'picp': picp,
        'mpiw': mpiw,
        'lower_bounds': y_pred_l_adjusted,
        'upper_bounds': y_pred_u_adjusted,
        'time': end_time - start_time
    }

def select_features(X, y, threshold=0.2):
    """Select important features based on correlation with target."""
    correlations = np.abs(np.array([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])]))
    selected_indices = np.where(correlations > threshold)[0]
    
    # Ensure we have at least a few features
    if len(selected_indices) < 2:
        # If no features meet the threshold, take the top 3
        top_indices = np.argsort(correlations)[-3:]
        return top_indices
        
    return selected_indices

def run_no2_experiment():
    """Run experiments on NO2 dataset before and after feature selection."""
    print("\n=========== Running Experiments on NO2 Dataset ===========")
    
    # Load dataset
    X, y, feature_names = load_no2_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Dataset shape: {X.shape}")
    
    # ------------------------
    # Before feature selection
    # ------------------------
    print("\n--- Before Feature Selection ---")
    
    # SVQR+CP
    print("Running SVQR+CP...")
    svqr_result = svqr_cp(X_train, y_train, X_test, y_test, alpha=0.1)
    print(f"SVQR+CP - PICP: {svqr_result['picp']*100:.2f}%, MPIW: {svqr_result['mpiw']:.4f}, Time: {svqr_result['time']:.2f}s")
    
    # SSVQR+CP
    print("Running SSVQR+CP...")
    ssvqr_result = svqr_cp(X_train, y_train, X_test, y_test, alpha=0.1, sparse=True)
    print(f"SSVQR+CP - PICP: {ssvqr_result['picp']*100:.2f}%, MPIW: {ssvqr_result['mpiw']:.4f}, Time: {ssvqr_result['time']:.2f}s")
    
    # ------------------------
    # Feature selection
    # ------------------------
    selected_indices = select_features(X, y, threshold=0.2)
    
    X_selected = X[:, selected_indices]
    X_train_selected = X_train[:, selected_indices]
    X_test_selected = X_test[:, selected_indices]
    
    print(f"\nFeature selection: Kept {len(selected_indices)}/{X.shape[1]} features ({len(selected_indices)/X.shape[1]*100:.1f}%)")
    print(f"Selected feature indices: {selected_indices}")
    
    # ------------------------
    # After feature selection
    # ------------------------
    print("\n--- After Feature Selection ---")
    
    # SVQR+CP
    print("Running SVQR+CP after feature selection...")
    svqr_result_after = svqr_cp(X_train_selected, y_train, X_test_selected, y_test, alpha=0.1)
    print(f"SVQR+CP - PICP: {svqr_result_after['picp']*100:.2f}%, MPIW: {svqr_result_after['mpiw']:.4f}, Time: {svqr_result_after['time']:.2f}s")
    
    # SSVQR+CP
    print("Running SSVQR+CP after feature selection...")
    ssvqr_result_after = svqr_cp(X_train_selected, y_train, X_test_selected, y_test, alpha=0.1, sparse=True)
    print(f"SSVQR+CP - PICP: {ssvqr_result_after['picp']*100:.2f}%, MPIW: {ssvqr_result_after['mpiw']:.4f}, Time: {ssvqr_result_after['time']:.2f}s")
    
    # Generate LaTeX table row
    retention = len(selected_indices) / X.shape[1] * 100
    
    print(f"\n--- LaTeX Table Row for NO2 ---")
    print("\\multirow{2}{*}{\\textbf{NO2}}")
    print(f"& SVQR+CP & {svqr_result['picp']*100:.2f} & {svqr_result['mpiw']:.2f} & {svqr_result['time']:.2f} & {svqr_result_after['picp']*100:.2f} & {svqr_result_after['mpiw']:.2f} & {svqr_result_after['time']:.2f} & \\multirow{{2}}{{*}}{{{retention:.1f}\\%}} \\\\")
    print(f"& SSVQR+CP & {ssvqr_result['picp']*100:.2f} & {ssvqr_result['mpiw']:.2f} & {ssvqr_result['time']:.2f} & {ssvqr_result_after['picp']*100:.2f} & {ssvqr_result_after['mpiw']:.2f} & {ssvqr_result_after['time']:.2f} & \\\\")

# Run the experiment
if __name__ == "__main__":
    run_no2_experiment() 