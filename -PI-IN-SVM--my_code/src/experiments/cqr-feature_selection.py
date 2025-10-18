# feature_selection.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import time

# import your CQR routine
from cqr import train_two_model_cqr_conformal


def main():
    plt.close('all')

    # %% Step 1: Load & preprocess the Boston Housing dataset
    data = pd.read_excel("bostonhousingdata.xlsx", header=0)
    data = data.apply(pd.to_numeric, errors='coerce')
    imputer = SimpleImputer(strategy="mean")
    data_imputed = pd.DataFrame(imputer.fit_transform(data))
    print(f"Data shape after imputation: {data_imputed.shape}")

    data = data_imputed.to_numpy()
    X = data[:, :-1]
    Y = data[:, -1]

    # %% Step 2: Split into Train / Calibration / Test (60% / 20% / 20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42
    )

    # reshape to (n_samples, 1) for compatibility
    y_train = y_train.reshape(-1, 1)
    y_cal = y_cal.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    # %% Parameters for CQR
    q_low, q_high = 0.025, 0.975
    alpha = 0.05

    # %% Step 3: Before Feature Selection using CQR
    print("\n=== Before Feature Selection ===")
    (
        time_before,
        picp_before,
        mpiw_before,
        low_cov_before,
        high_cov_before,
        model_lo,
        model_hi,
        lb_before,
        ub_before
    ) = train_two_model_cqr_conformal(
        X_train, y_train, X_cal, y_cal, X_test, y_test,
        q_low=q_low, q_high=q_high, alpha=alpha,
        verbose=False
    )
    print(f"PICP: {picp_before:.4f}, MPIW: {mpiw_before:.4f}, Time: {time_before:.2f}s")

    # %% Step 4: Manual Feature Drop (hard‐coded)
    # We know from previous runs to drop features 3,4,7,8,10:
    keep_idx    = np.array([0, 1, 2, 5, 6, 9, 11, 12])
    dropped_idx = np.array([3, 4, 7, 8, 10])
    print("\nUsing manual feature drop:")
    print(f"  Dropped feature indices  : {dropped_idx.tolist()}")
    print(f"  Retained feature indices : {keep_idx.tolist()}")

    # %% Step 5: After Feature Selection using the same CQR routine
    print("\n=== After Feature Selection ===")
    X_train_sel = X_train[:, keep_idx]
    X_cal_sel   = X_cal[:, keep_idx]
    X_test_sel  = X_test[:, keep_idx]

    (
        time_after,
        picp_after,
        mpiw_after,
        low_cov_after,
        high_cov_after,
        model_lo_sel,
        model_hi_sel,
        lb_after,
        ub_after
    ) = train_two_model_cqr_conformal(
        X_train_sel, y_train, X_cal_sel, y_cal, X_test_sel, y_test,
        q_low=q_low, q_high=q_high, alpha=alpha,
        verbose=False
    )
    print(f"PICP: {picp_after:.4f}, MPIW: {mpiw_after:.4f}, Time: {time_after:.2f}s")

    # %% Step 6: Plotting Feature Importances
    # Compute full‐model importances (sum of abs input weights)
    W_lo_full = model_lo.layers[0].get_weights()[0]
    W_hi_full = model_hi.layers[0].get_weights()[0]
    imp_lo_full = np.sum(np.abs(W_lo_full), axis=1)
    imp_hi_full = np.sum(np.abs(W_hi_full), axis=1)

    # Plot before FS
    plt.figure()
    plt.bar(np.arange(1, imp_hi_full.size+1), imp_hi_full)
    plt.title("Feature Importances (Upper Quantile) Before FS")
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.grid(True)
    plt.xticks(np.arange(1, imp_hi_full.size+1))

    plt.figure()
    plt.bar(np.arange(1, imp_lo_full.size+1), imp_lo_full)
    plt.title("Feature Importances (Lower Quantile) Before FS")
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.grid(True)
    plt.xticks(np.arange(1, imp_lo_full.size+1))

    # Compute selected‐feature importances (after FS)
    W_lo_sel = model_lo_sel.layers[0].get_weights()[0]
    W_hi_sel = model_hi_sel.layers[0].get_weights()[0]
    imp_lo_sel = np.sum(np.abs(W_lo_sel), axis=1)
    imp_hi_sel = np.sum(np.abs(W_hi_sel), axis=1)

    # Plot after FS
    plt.figure()
    plt.bar(np.arange(1, imp_hi_sel.size+1), imp_hi_sel)
    plt.title("Feature Importances (Upper Quantile) After FS")
    plt.xlabel("Selected Feature Rank")
    plt.ylabel("Importance")
    plt.grid(True)
    plt.xticks(np.arange(1, imp_hi_sel.size+1), (keep_idx+1).tolist(), rotation=90)

    plt.figure()
    plt.bar(np.arange(1, imp_lo_sel.size+1), imp_lo_sel)
    plt.title("Feature Importances (Lower Quantile) After FS")
    plt.xlabel("Selected Feature Rank")
    plt.ylabel("Importance")
    plt.grid(True)
    plt.xticks(np.arange(1, imp_lo_sel.size+1), (keep_idx+1).tolist(), rotation=90)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()