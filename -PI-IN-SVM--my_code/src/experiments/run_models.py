#!/usr/bin/env python3
import subprocess
import os
import re
import pandas as pd
import datetime

# Create a directory to store results
results_dir = "model_results"
os.makedirs(results_dir, exist_ok=True)

# Timestamp for the run
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Function to extract metrics from output text
def extract_metrics(output_text):
    metrics = {}
    
    # For testandvalidation.py (SVR model)
    picp_match = re.search(r"Test set evaluation: PICP = (\d+\.\d+), MPIW = (\d+\.\d+)", output_text)
    if picp_match:
        metrics["SVR_PICP"] = float(picp_match.group(1))
        metrics["SVR_MPIW"] = float(picp_match.group(2))
    
    exec_time_match = re.search(r"Program execution complete in (\d+\.\d+) seconds", output_text)
    if exec_time_match:
        metrics["SVR_execution_time"] = float(exec_time_match.group(1))
    
    # For cqr.py (CQR model)
    cqr_mean_match = re.search(r"Mean PICP: (\d+\.\d+) ± (\d+\.\d+)", output_text)
    if cqr_mean_match:
        metrics["CQR_mean_PICP"] = float(cqr_mean_match.group(1))
        metrics["CQR_std_PICP"] = float(cqr_mean_match.group(2))
    
    mpiw_match = re.search(r"Mean MPIW: (\d+\.\d+) ± (\d+\.\d+)", output_text)
    if mpiw_match:
        metrics["CQR_mean_MPIW"] = float(mpiw_match.group(1))
        metrics["CQR_std_MPIW"] = float(mpiw_match.group(2))
    
    train_time_match = re.search(r"Mean Training Time: (\d+\.\d+) seconds", output_text)
    if train_time_match:
        metrics["CQR_mean_training_time"] = float(train_time_match.group(1))
    
    return metrics

# Run SVR model (testandvalidation.py)
print("Running SVR model (testandvalidation.py)...")
svr_output = subprocess.run(["python", "testandvalidation.py"], 
                            capture_output=True, text=True)
svr_stdout = svr_output.stdout
svr_stderr = svr_output.stderr

# Save SVR output
with open(f"{results_dir}/svr_output_{timestamp}.txt", "w") as f:
    f.write(svr_stdout)
    if svr_stderr:
        f.write("\n--- ERRORS ---\n")
        f.write(svr_stderr)

# Run CQR model (cqr.py)
print("Running CQR model (cqr.py)...")
cqr_output = subprocess.run(["python", "cqr.py"], 
                           capture_output=True, text=True)
cqr_stdout = cqr_output.stdout
cqr_stderr = cqr_output.stderr

# Save CQR output
with open(f"{results_dir}/cqr_output_{timestamp}.txt", "w") as f:
    f.write(cqr_stdout)
    if cqr_stderr:
        f.write("\n--- ERRORS ---\n")
        f.write(cqr_stderr)

# Extract metrics
svr_metrics = extract_metrics(svr_stdout)
cqr_metrics = extract_metrics(cqr_stdout)

# Combine all metrics
all_metrics = {**svr_metrics, **cqr_metrics, "timestamp": timestamp}

# Create or update the CSV file with results
results_file = f"{results_dir}/model_comparison_results.csv"
if os.path.exists(results_file):
    results_df = pd.read_csv(results_file)
    results_df = pd.concat([results_df, pd.DataFrame([all_metrics])], ignore_index=True)
else:
    results_df = pd.DataFrame([all_metrics])

results_df.to_csv(results_file, index=False)

# Also save the current run results separately
current_run_df = pd.DataFrame([all_metrics])
current_run_df.to_csv(f"{results_dir}/results_{timestamp}.csv", index=False)

print(f"Results saved to {results_dir}/")
print("Summary of results:")
for k, v in all_metrics.items():
    if k != "timestamp":
        print(f"{k}: {v}") 