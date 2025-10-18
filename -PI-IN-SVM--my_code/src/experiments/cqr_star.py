import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import cqr
from cqr import train_two_model_cqr_conformal

def load_star_data(base_path='.'):
    # load STAR.csv from the provided base_path (defaults to '.')
    df = pd.read_csv(f"STAR.csv")

    # --- Categorical → numeric mappings ---
    df.loc[df['gender']=='female','gender'] = 0
    df.loc[df['gender']=='male',  'gender'] = 1

    eth_map = {'cauc':0,'afam':1,'asian':2,'hispanic':3,'amindian':4,'other':5}
    df['ethnicity'] = df['ethnicity'].map(eth_map)

    star_map = {'regular':0,'small':1,'regular+aide':2}
    for col in ['stark','star1','star2','star3']:
        df[col] = df[col].map(star_map)

    lunch_map = {'free':0,'non-free':1}
    for col in ['lunchk','lunch1','lunch2','lunch3']:
        df[col] = df[col].map(lunch_map)

    school_map = {'inner-city':0,'suburban':1,'rural':2,'urban':3}
    for col in ['schoolk','school1','school2','school3']:
        df[col] = df[col].map(school_map)

    df.loc[df['degreek']=='bachelor','degreek'] = 0
    df.loc[df['degreek']=='master',   'degreek'] = 1
    df.loc[df['degreek']=='specialist','degreek'] = 2
    df.loc[df['degreek']=='master+',  'degreek'] = 3
    for col in ['degree1','degree2','degree3']:
        df.loc[df[col]=='bachelor',col] = 0
        df.loc[df[col]=='master',   col] = 1
        df.loc[df[col]=='specialist',col] = 2
        df.loc[df[col]=='phd',      col] = 3

    ladder_map = {'level1':0,'level2':1,'level3':2,'apprentice':3,
                  'probation':4,'pending':5,'noladder':5,'notladder':6}
    for col in ['ladderk','ladder1','ladder2','ladder3']:
        df[col] = df[col].map(ladder_map)

    te_map = {'cauc':0,'afam':1,'asian':2}
    for col in ['tethnicityk','tethnicity1','tethnicity2','tethnicity3']:
        df[col] = df[col].map(te_map)

    # Drop any incomplete rows
    df = df.dropna()

    # Build target = sum of reading & math across K–3
    grade = (
        df["readk"] + df["read1"] + df["read2"] + df["read3"]
      + df["mathk"] + df["math1"] + df["math2"] + df["math3"]
    )

    # Features = all columns except the 8 read/math columns & lunchk
    names = df.columns.values
    data_names = np.concatenate((names[:8], names[17:]))
    X = df.loc[:, data_names].values.astype(np.float32)
    y = grade.values.astype(np.float32)

    return X, y

if __name__ == "__main__":
    # Load and standardize
    X, y = load_star_data()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Conformal parameters
    q_low, q_high, alpha = 0.025, 0.975, 0.05
    verbose = True

    coverages = []; mpiws = []; training_times = []
    low_covs = []; up_covs = []

    print("Running CQR model on STAR dataset 10 times...")
    global run_counter
    run_counter = 0
    cqr.run_counter = run_counter

    for i in range(10):
        run_counter = i
        cqr.run_counter = run_counter
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_scaled, y, test_size=0.20, random_state=42+i
        )
        X_train, X_cal, y_train, y_cal = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42+i
        )

        # reshape for TF
        y_train = y_train.reshape(-1,1)
        y_cal   = y_cal.reshape(-1,1)
        y_test  = y_test.reshape(-1,1)

        results = train_two_model_cqr_conformal(
            X_train, y_train, X_cal, y_cal, X_test, y_test,
            q_low=q_low, q_high=q_high, alpha=alpha, verbose=verbose
        )
        t, picp, mw, lowc, upc, *_ = results
        training_times.append(t)
        coverages.append(picp)
        mpiws.append(mw)
        low_covs.append(lowc)
        up_covs.append(upc)

    # Summary stats
    mean_cov = np.mean(coverages); std_cov = np.std(coverages)
    mean_mpiw = np.mean(mpiws); std_mpiw = np.std(mpiws)
    mean_time = np.mean(training_times)
    mean_low = np.mean(low_covs); mean_up = np.mean(up_covs)

    print("\n=== Summary over 10 runs ===")
    print(f"Mean PICP: {mean_cov:.4f} ± {std_cov:.4f}")
    print(f"Mean MPIW: {mean_mpiw:.4f} ± {std_mpiw:.4f}")
    print(f"Mean LowCov: {mean_low:.4f}, Mean UpCov: {mean_up:.4f}")
    print(f"Mean Training Time: {mean_time:.2f} seconds")

    # Plot last run intervals
    plt.figure(figsize=(10,6))
    y_test_1d = y_test.flatten()
    idx = np.argsort(y_test_1d)
    yt, lb, ub = y_test_1d[idx], results[-2][idx], results[-1][idx]
    x = np.arange(len(yt))
    plt.scatter(x, yt, color='blue', alpha=0.6, label='Actual')
    plt.fill_between(x, lb, ub, color='gray', alpha=0.3,
                     label=f'Interval ({1-alpha:.0%})')
    plt.title(f"STAR Dataset Conformal CQR\nPICP {mean_cov:.4f}±{std_cov:.4f}, MPIW {mean_mpiw:.4f}±{std_mpiw:.4f}")
    plt.xlabel("Sample index (sorted)")
    plt.ylabel("Total grade (reading + math)")
    plt.legend()
    plt.tight_layout()
    plt.show() 