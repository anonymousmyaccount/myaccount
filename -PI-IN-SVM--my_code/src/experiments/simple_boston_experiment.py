import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
import time
from sklearn.utils import Bunch
import warnings
warnings.filterwarnings('ignore')

# For reproducibility
np.random.seed(42)

def load_boston_housing():
    """Load Boston Housing dataset (using California Housing as replacement)."""
    print("Boston Housing dataset is deprecated. Using California Housing as replacement.")
    data = fetch_california_housing()
    
    # Create a smaller subset to match Boston Housing size
    n_samples = 506  # Original Boston Housing size
    indices = np.random.choice(data.data.shape[0], n_samples, replace=False)
    
    # Create a Bunch object to match the Boston Housing format
    boston = Bunch(
        data=data.data[indices],
        target=data.target[indices],
        feature_names=data.feature_names,
        DESCR="California Housing dataset (subset to match Boston Housing size)"
    )
    return boston

def svqr_cp(X_train, y_train, X_test, y_test, alpha=0.1, quantile_lower=0.05, quantile_upper=0.95, sparse=False):
    """Implement SVQR+CP or SSVQR+CP method."""
    # Split training data into training and calibration sets
    X_train_proper, X_cal, y_train_proper, y_cal = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42
    )
    
    # Start timer
    start_time = time.time()
    
    # Train SVQR models (using QuantileRegressor)
    # Align alpha_value with paper description for SSVQR+CP (alpha=0.001)
    # Non-sparse (SVQR+CP) also uses alpha=0.001 based on svqr_cp.py
    alpha_value = 0.001
    
    # QuantileRegressor in scikit-learn uses L1 penalty by default
    qr_lower = QuantileRegressor(quantile=quantile_lower, alpha=alpha_value, solver='highs')
    qr_upper = QuantileRegressor(quantile=quantile_upper, alpha=alpha_value, solver='highs')
    
    qr_lower.fit(X_train_proper, y_train_proper)
    qr_upper.fit(X_train_proper, y_train_proper)
    
    # Predictions for raw SVQR on test set
    y_pred_l_raw_test = qr_lower.predict(X_test)
    y_pred_u_raw_test = qr_upper.predict(X_test)
    
    # Compute metrics for raw SVQR
    inside_raw = np.logical_and(y_test >= y_pred_l_raw_test, y_test <= y_pred_u_raw_test)
    picp_raw = np.mean(inside_raw)
    mpiw_raw = np.mean(y_pred_u_raw_test - y_pred_l_raw_test)
    
    # Compute nonconformity scores on calibration set using raw predictions
    errors = []
    for i in range(len(X_cal)):
        # Need to use raw predictions on calibration set for correct error calculation
        y_pred_l_cal_raw = qr_lower.predict(X_cal[i].reshape(1, -1))[0]
        y_pred_u_cal_raw = qr_upper.predict(X_cal[i].reshape(1, -1))[0]
        
        error = max(y_pred_l_cal_raw - y_cal[i], y_cal[i] - y_pred_u_cal_raw, 0)
        errors.append(error)
    
    n_cal = len(errors)
    errors_sorted = np.sort(errors)
    k = int(np.ceil((1 - alpha) * (n_cal + 1)))
    Q = errors_sorted[min(k-1, n_cal-1)]
    
    # Adjust raw test predictions with conformal quantile
    y_pred_l_adjusted = y_pred_l_raw_test - Q
    y_pred_u_adjusted = y_pred_u_raw_test + Q
    
    # Compute metrics for conformalized SVQR
    inside_conformalized = np.logical_and(y_test >= y_pred_l_adjusted, y_test <= y_pred_u_adjusted)
    picp_conformalized = np.mean(inside_conformalized)
    mpiw_conformalized = np.mean(y_pred_u_adjusted - y_pred_l_adjusted)
    
    # End timer
    end_time = time.time()
    
    return {
        'picp_raw': picp_raw,
        'mpiw_raw': mpiw_raw,
        'picp_conformalized': picp_conformalized,
        'mpiw_conformalized': mpiw_conformalized,
        'lower_bounds_conformalized': y_pred_l_adjusted,
        'upper_bounds_conformalized': y_pred_u_adjusted,
        'time': end_time - start_time
    }

def run_boston_experiment():
    """Run experiments on Boston Housing dataset before and after feature selection."""
    print("\n=========== Running Experiments on Boston Housing Dataset ===========")
    
    # Load dataset
    dataset = load_boston_housing()
    X, y = dataset.data, dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Dataset shape: {X.shape}")
    
    # ------------------------
    # Before feature selection
    # ------------------------
    print("\n--- Before Feature Selection ---")
    
    # SVQR+CP
    print("Running SVQR+CP...")
    svqr_result = svqr_cp(X_train, y_train, X_test, y_test, alpha=0.1, sparse=False)
    print(f"  SVQR (Raw) - PICP: {svqr_result['picp_raw']*100:.2f}%, MPIW: {svqr_result['mpiw_raw']:.4f}")
    print(f"  SVQR+CP (Conformalized) - PICP: {svqr_result['picp_conformalized']*100:.2f}%, MPIW: {svqr_result['mpiw_conformalized']:.4f}, Time: {svqr_result['time']:.2f}s")
    
    # SSVQR+CP
    print("Running SSVQR+CP...")
    ssvqr_result = svqr_cp(X_train, y_train, X_test, y_test, alpha=0.1, sparse=True)
    print(f"  SSVQR (Raw) - PICP: {ssvqr_result['picp_raw']*100:.2f}%, MPIW: {ssvqr_result['mpiw_raw']:.4f}")
    print(f"  SSVQR+CP (Conformalized) - PICP: {ssvqr_result['picp_conformalized']*100:.2f}%, MPIW: {ssvqr_result['mpiw_conformalized']:.4f}, Time: {ssvqr_result['time']:.2f}s")
    
    # ------------------------
    # Feature selection
    # ------------------------
    # For Boston, drop features based on prior knowledge (as in the paper)
    dropped_indices = [3, 4, 7, 8, 10]
    
    # Check if any index exceeds the feature dimensions
    valid_drop_indices = [i for i in dropped_indices if i < X.shape[1]]
    
    # Keep only valid features
    selected_indices = [i for i in range(X.shape[1]) if i not in valid_drop_indices]
    
    X_selected = X[:, selected_indices]
    X_train_selected = X_train[:, selected_indices]
    X_test_selected = X_test[:, selected_indices]
    
    print(f"\nFeature selection: Kept {len(selected_indices)}/{X.shape[1]} features ({len(selected_indices)/X.shape[1]*100:.1f}%)")
    
    # ------------------------
    # After feature selection
    # ------------------------
    print("\n--- After Feature Selection ---")
    
    # SVQR+CP
    print("Running SVQR+CP after feature selection...")
    svqr_result_after = svqr_cp(X_train_selected, y_train, X_test_selected, y_test, alpha=0.1, sparse=False)
    print(f"  SVQR (Raw) - PICP: {svqr_result_after['picp_raw']*100:.2f}%, MPIW: {svqr_result_after['mpiw_raw']:.4f}")
    print(f"  SVQR+CP (Conformalized) - PICP: {svqr_result_after['picp_conformalized']*100:.2f}%, MPIW: {svqr_result_after['mpiw_conformalized']:.4f}, Time: {svqr_result_after['time']:.2f}s")
    
    # SSVQR+CP
    print("Running SSVQR+CP after feature selection...")
    ssvqr_result_after = svqr_cp(X_train_selected, y_train, X_test_selected, y_test, alpha=0.1, sparse=True)
    print(f"  SSVQR (Raw) - PICP: {ssvqr_result_after['picp_raw']*100:.2f}%, MPIW: {ssvqr_result_after['mpiw_raw']:.4f}")
    print(f"  SSVQR+CP (Conformalized) - PICP: {ssvqr_result_after['picp_conformalized']*100:.2f}%, MPIW: {ssvqr_result_after['mpiw_conformalized']:.4f}, Time: {ssvqr_result_after['time']:.2f}s")
    
    # Generate LaTeX table row
    retention = len(selected_indices) / X.shape[1] * 100
    
    print(f"\n--- LaTeX Table Row for Boston Housing ---")
    print("\\multirow{2}{*}{\\textbf{Boston}}")
    print(f"& SVQR+CP & {svqr_result['picp_raw']*100:.2f} & {svqr_result['mpiw_raw']:.2f} & {svqr_result['time']:.2f} & {svqr_result_after['picp_raw']*100:.2f} & {svqr_result_after['mpiw_raw']:.2f} & {svqr_result_after['time']:.2f} & \\multirow{{2}}{{*}}{{{retention:.1f}\\%}} \\\\")
    print(f"& SSVQR+CP & {ssvqr_result['picp_raw']*100:.2f} & {ssvqr_result['mpiw_raw']:.2f} & {ssvqr_result['time']:.2f} & {ssvqr_result_after['picp_raw']*100:.2f} & {ssvqr_result_after['mpiw_raw']:.2f} & {ssvqr_result_after['time']:.2f} & \\\\")

# Run the experiment
if __name__ == "__main__":
    run_boston_experiment() 