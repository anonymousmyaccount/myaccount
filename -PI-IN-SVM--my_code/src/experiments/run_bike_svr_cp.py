import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split

# Dynamically load lp_svr12 module
import importlib.util
spec = importlib.util.spec_from_file_location("lp_svr12", "lp_svr12.py")
lp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lp)
# Disable verbose printing in lp_svr12
lp.verbose = False
quantileSVR = lp.quantileLPONENORMTSVR12
evaluate_PICP_MPIW = lp.evaluate_PICP

# Import feature selection function from MADELON
spec_fs = importlib.util.spec_from_file_location("fs", "MADELON/feature_selection.py")
fs = importlib.util.module_from_spec(spec_fs)
spec_fs.loader.exec_module(fs)
linearTSVR = fs.linear_quantile_lponenorm_tsvr

# -------------------------------
# 1. Load and preprocess bike data
# -------------------------------
df = pd.read_csv("../datasets/bike_train.csv")
print(f"[1] Loaded bike data: {df.shape[0]} samples, {df.shape[1]-1} features (excluding target)")
# Season and weather dummies
df = pd.concat([df, pd.get_dummies(df['season'], prefix='season')], axis=1)
df = pd.concat([df, pd.get_dummies(df['weather'], prefix='weather')], axis=1)
df.drop(['season','weather'], axis=1, inplace=True)
# Extract datetime features
df['hour'] = pd.to_datetime(df['datetime']).dt.hour
df['day']  = pd.to_datetime(df['datetime']).dt.dayofweek
df['month'] = pd.to_datetime(df['datetime']).dt.month
df['year'] = pd.to_datetime(df['datetime']).dt.year.map({2011:0,2012:1})
# Drop unused cols
df.drop(['datetime','casual','registered'], axis=1, inplace=True)
# Features and target
y = df['count'].values.reshape(-1,1)
X = df.drop('count', axis=1).values.astype(np.float64)
print(f"[2] Final feature matrix X shape: {X.shape}, target vector y shape: {y.shape}")

# -------------------------------
# 2. Split: 20% test, then 70/30 train/calibration
# -------------------------------
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_train, X_cal, y_train, y_cal = train_test_split(X_temp, y_temp, test_size=0.30, random_state=42)
print(f"[3] Split data -> Train: {X_train.shape[0]} samples, Cal: {X_cal.shape[0]} samples, Test: {X_test.shape[0]} samples")

# -------------------------------
# Hyperparameters for Feature Selection (and later model training)
# -------------------------------
s = 0.1      # RBF gamma (Note: linearTSVR is linear, so this 's' for RBF is not used by it but defined here for consistency if switching to RBF later)
c1 = 1.0     # pinball penalty
c3 = 1.0     # SVQR regularization (used as C3 in linearTSVR)
tau_lower = 0.05 # Define tau_lower here as it's used in feature selection
tau_upper = 0.95 # Define tau_upper here as it's used in feature selection
print(f"[Pre-FS] Hyperparameters -> gamma_rbf (s): {s}, C3 (SVR reg): {c3}, C1 (pinball): {c1}, tau: ({tau_lower}, {tau_upper})")

# -------------------------------
# Feature Selection
# -------------------------------
print("[4] Performing feature selection using LP-SVR weights...")
# Compute lower quantile weights
# linearTSVR is a linear model; 's' (gamma for RBF) is not used. Pass a placeholder if needed, or ensure it handles it.
# Assuming s, c3, c1 are the intended parameters for the linear model as well, based on context.
_, _, spar_low, w_low = linearTSVR(X_train, y_train.flatten(), X_cal, 0, c3, c1, tau_lower) # Pass 0 or some default for 's' as it's linear
# Compute upper quantile weights
_, _, spar_up, w_up = linearTSVR(X_train, y_train.flatten(), X_cal, 0, c3, c1, tau_upper) # Pass 0 or some default for 's'
# Exclude intercept (last entry)
n_feats = X_train.shape[1]
w_low_feat = np.abs(w_low[:n_feats])
w_up_feat = np.abs(w_up[:n_feats])
# Thresholding small weights
thr = 1e-4
valid = np.where((w_low_feat >= thr) | (w_up_feat >= thr))[0]
print(f"    Original features: {n_feats}, Retained: {len(valid)}, Dropped: {n_feats - len(valid)}")
# Filter datasets
X_train = X_train[:, valid]
X_cal   = X_cal[:, valid]
X_test  = X_test[:, valid]
print(f"    Filtered feature dimension: {X_train.shape[1]}")

# -------------------------------
# 3. Hyperparameters and quantiles (Model Training Post-FS)
#    (Re-state or confirm parameters. 's' for RBF kernel will be used by quantileSVR)
# -------------------------------
# s = 0.1 is already defined above for RBF, used by quantileSVR (non-linear)
# c1 = 1.0 and c3 = 1.0 are also defined
# tau_lower = 0.05 and tau_upper = 0.95 are also defined
print(f"[Post-FS Model Training] Hyperparameters -> RBF gamma (s): {s}, C3 (SVR reg): {c3}, C1 (pinball): {c1}")

# -------------------------------
# 4. Compute quantile predictions on calibration set (using RBF SVR)
# -------------------------------
print("[5] Computing lower quantile on CALIBRATION set (post-FS, RBF SVR)...")
start_time = time.time()
# Ensure y_train is flattened for quantileSVR if it expects 1D y
_, lower_cal, _ = quantileSVR(X_train, y_train.flatten(), X_cal, s, c3, c1, tau_lower)
print(f"    Lower calibration quantiles shape: {lower_cal.shape}")
print("[6] Computing upper quantile on CALIBRATION set (post-FS, RBF SVR)...")
_, upper_cal, _ = quantileSVR(X_train, y_train.flatten(), X_cal, s, c3, c1, tau_upper)
print(f"    Upper calibration quantiles shape: {upper_cal.shape}")

# Conformal calibration: nonconformity scores
errors = np.maximum(lower_cal - y_cal.flatten(), y_cal.flatten() - upper_cal)
n_cal = len(errors)
k = int(np.ceil((1-0.1)*(n_cal+1)))
Q = np.sort(errors)[min(k-1, n_cal-1)]
cal_time = time.time() - start_time
print(f"[7] Calibration complete in {cal_time:.2f}s. Q (90% nonconformity): {Q:.4f}")

# -------------------------------
# 5. Apply on test set and adjust intervals (using RBF SVR)
# -------------------------------
print("[8] Computing lower quantile on TEST set (post-FS, RBF SVR)...")
start_time = time.time()
_, lower_test, _ = quantileSVR(X_train, y_train.flatten(), X_test, s, c3, c1, tau_lower)
print("[9] Computing upper quantile on TEST set (post-FS, RBF SVR)...")
_, upper_test, _ = quantileSVR(X_train, y_train.flatten(), X_test, s, c3, c1, tau_upper)

# Adjust with conformal quantile
lower_test_adj = lower_test - Q
upper_test_adj = upper_test + Q
print(f"[10] Test adjustments: applied Q to widen intervals.")

picp, mpiw = evaluate_PICP_MPIW(y_test, lower_test_adj, upper_test_adj)
test_time = time.time() - start_time

# -------------------------------
# 6. Print results
# -------------------------------
print(f"Calibration time: {cal_time:.2f}s, Test prediction time: {test_time:.2f}s")
print(f"[11] Final Test Metrics -> PICP: {picp*100:.2f}%, MPIW: {mpiw:.4f}") 