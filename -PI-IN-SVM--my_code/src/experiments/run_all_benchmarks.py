import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import time
import os
import warnings

# Import functions from the models module
from src.models.quantile_svr import svqr_cp, cqr_nn, load_boston_housing, load_energy_efficiency

# Path for saving figures (relative to this script's location in src/experiments)
RESULTS_DIR = '../../results_new/figures'
os.makedirs(RESULTS_DIR, exist_ok=True)

# For reproducibility - Note: svqr_cp also has internal seeds.
# torch.manual_seed(42) # Seed is set per run in CQR-NN

def run_all_experiments(n_runs=5):
    """Run all experiments and collect results."""
    results = {}
    
    # Datasets to test
    datasets = [
        ('Boston Housing', load_boston_housing()),
        ('Energy Efficiency', load_energy_efficiency())
    ]
    
    for dataset_name, dataset in datasets:
        print(f"\n{'-'*60}\nRunning experiments on {dataset_name} dataset\n{'-'*60}")
        
        X, y = dataset.data, dataset.target
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize results structure
        results[dataset_name] = {
            'SVQR+CP': {'picp': [], 'mpiw': []},
            'CQR-NN': {'picp': [], 'mpiw': []}
        }
        
        # Run SVQR+CP (just once as it's deterministic)
        print("Running SVQR+CP...")
        start_time_svqr = time.time()
        svqr_results = svqr_cp(X_train, y_train, X_test, y_test)
        end_time_svqr = time.time()
        
        print(f"SVQR+CP - PICP: {svqr_results['picp']:.4f}, MPIW: {svqr_results['mpiw']:.4f}")
        print(f"Time taken: {end_time_svqr - start_time_svqr:.2f} seconds")
        
        results[dataset_name]['SVQR+CP']['picp'].append(svqr_results['picp'])
        results[dataset_name]['SVQR+CP']['mpiw'].append(svqr_results['mpiw'])
        results[dataset_name]['SVQR+CP']['time'] = end_time_svqr - start_time_svqr # Store time
        
        # Run CQR-NN multiple times with different random seeds
        print(f"\nRunning CQR-NN ({n_runs} runs)...")
        cqr_nn_times = []
        # To store the outputs of the last CQR-NN run for visualization
        last_cqr_run_results = None 
        for run_idx in range(n_runs):
            print(f"Run {run_idx+1}/{n_runs}")
            # Set different random seed for each run
            torch.manual_seed(42 + run_idx) # Ensure reproducibility for each run
            
            start_time_cqr = time.time()
            current_cqr_results = cqr_nn(X_train, y_train, X_test, y_test) 
            end_time_cqr = time.time()
            cqr_nn_times.append(end_time_cqr - start_time_cqr)
            last_cqr_run_results = current_cqr_results # Keep track of the last run
            
            print(f"CQR-NN - PICP: {current_cqr_results['picp']:.4f}, MPIW: {current_cqr_results['mpiw']:.4f}")
            print(f"Time taken: {end_time_cqr - start_time_cqr:.2f} seconds")
            
            results[dataset_name]['CQR-NN']['picp'].append(current_cqr_results['picp'])
            results[dataset_name]['CQR-NN']['mpiw'].append(current_cqr_results['mpiw'])
        
        results[dataset_name]['CQR-NN']['time_avg'] = np.mean(cqr_nn_times) if cqr_nn_times else float('nan')
        results[dataset_name]['CQR-NN']['time_std'] = np.std(cqr_nn_times) if cqr_nn_times else float('nan')

        # Create visualization for this dataset using the results from the last CQR-NN run for plotting intervals
        if last_cqr_run_results is not None:
            create_visualization(dataset_name, results[dataset_name], y_test, 
                                 svqr_results, last_cqr_run_results) 
    
    # Print summary of results
    print_summary(results)
    
    # TODO: Add markdown output generation here if needed, e.g., to results_new/experimental_results.md
    # For now, it prints to console.

    return results

def create_visualization(dataset_name, current_dataset_results_summary, y_test, svqr_interval_data, cqr_interval_data):
    """Create visualization of prediction intervals."""
    # Create figure for PICP and MPIW comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot PICP
    methods = ['SVQR+CP', 'CQR-NN']
    picp_svqr_vals = current_dataset_results_summary['SVQR+CP']['picp']
    picp_cqr_vals = current_dataset_results_summary['CQR-NN']['picp']

    picp_vals = [
        np.mean(picp_svqr_vals) if len(picp_svqr_vals) > 0 else float('nan'), 
        np.mean(picp_cqr_vals) if len(picp_cqr_vals) > 0 else float('nan')
    ]
    picp_std = [
        np.std(picp_svqr_vals) if len(picp_svqr_vals) > 0 else float('nan'), 
        np.std(picp_cqr_vals) if len(picp_cqr_vals) > 0 else float('nan')
    ]
    
    ax1.bar(methods, picp_vals, yerr=picp_std, capsize=5, color=['blue', 'green'])
    ax1.axhline(y=0.9, color='r', linestyle='--', label='Nominal 90%')
    ax1.set_ylabel('PICP')
    ax1.set_title('Prediction Interval Coverage Probability')
    # Dynamic Y-lim, handle empty or NaN picp_vals for min/max
    valid_picp_vals = [v for v in picp_vals if not np.isnan(v)]
    min_y_lim = min(0.8, min(valid_picp_vals)-0.05 if len(valid_picp_vals) > 0 else 0.8)
    max_y_lim = max(1.0, max(valid_picp_vals)+0.05 if len(valid_picp_vals) > 0 else 1.0)
    ax1.set_ylim(min_y_lim, max_y_lim)
    ax1.legend()
    
    # Plot MPIW
    mpiw_svqr_vals = current_dataset_results_summary['SVQR+CP']['mpiw']
    mpiw_cqr_vals = current_dataset_results_summary['CQR-NN']['mpiw']
    mpiw_vals = [
        np.mean(mpiw_svqr_vals) if len(mpiw_svqr_vals) > 0 else float('nan'), 
        np.mean(mpiw_cqr_vals) if len(mpiw_cqr_vals) > 0 else float('nan')
    ]
    mpiw_std = [
        np.std(mpiw_svqr_vals) if len(mpiw_svqr_vals) > 0 else float('nan'), 
        np.std(mpiw_cqr_vals) if len(mpiw_cqr_vals) > 0 else float('nan')
    ]
    
    ax2.bar(methods, mpiw_vals, yerr=mpiw_std, capsize=5, color=['blue', 'green'])
    ax2.set_ylabel('MPIW')
    ax2.set_title('Mean Prediction Interval Width')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{dataset_name.replace(" ", "_")}_comparison.png'))
    plt.close(fig) # Close figure to free memory
    
    # Create a sample of prediction intervals
    if not (svqr_interval_data and cqr_interval_data and 'lower_bounds' in svqr_interval_data and 'lower_bounds' in cqr_interval_data):
        print(f"Skipping interval plot for {dataset_name} due to missing interval data.")
        return
        
    plt.figure(figsize=(10, 6))
    
    # Sample 100 points to plot or all if less than 100
    num_samples_to_plot = min(100, len(y_test))
    if num_samples_to_plot == 0:
        print(f"Skipping interval plot for {dataset_name}: no test samples.")
        plt.close() # Close the figure if created
        return
        
    indices = np.arange(len(y_test))
    np.random.shuffle(indices) # Shuffle for random sampling
    sample_indices = indices[:num_samples_to_plot]
    
    # Sort by target value for better visualization
    sort_idx = np.argsort(y_test[sample_indices])
    sorted_indices = sample_indices[sort_idx]
    
    # Get sample data
    y_sample = y_test[sorted_indices]
    x_points = np.arange(len(y_sample))
    
    # Plot SVQR+CP intervals
    lower_svqr = svqr_interval_data['lower_bounds'][sorted_indices]
    upper_svqr = svqr_interval_data['upper_bounds'][sorted_indices]
    
    plt.fill_between(x_points, lower_svqr, upper_svqr, alpha=0.3, color='blue', label='SVQR+CP Interval')
    
    # Plot CQR-NN intervals (using the interval data passed, e.g., from the last run)
    lower_cqr = cqr_interval_data['lower_bounds'][sorted_indices]
    upper_cqr = cqr_interval_data['upper_bounds'][sorted_indices]
    
    plt.fill_between(x_points, lower_cqr, upper_cqr, alpha=0.3, color='green', label='CQR-NN Interval')
    
    # Plot actual values
    plt.scatter(x_points, y_sample, color='red', s=20, label='Actual')
    
    plt.legend()
    plt.title(f'Prediction Intervals for {dataset_name} (Sample of {num_samples_to_plot} points)')
    plt.xlabel('Sample Index (sorted by target value)')
    plt.ylabel('Target Value')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{dataset_name.replace(" ", "_")}_intervals.png'))
    plt.close() # Close figure

def print_summary(results_summary_data):
    """Print a summary of all results."""
    summary_lines = ["\n" + "="*80, "SUMMARY OF RESULTS (from run_all_benchmarks.py)", "="*80]
    
    for dataset_name, dataset_results in results_summary_data.items():
        summary_lines.append(f"\n{dataset_name} Dataset:")
        summary_lines.append("-" * 50)
        
        # SVQR+CP results
        svqr_picp_list = dataset_results['SVQR+CP']['picp']
        svqr_mpiw_list = dataset_results['SVQR+CP']['mpiw']
        svqr_picp = np.mean(svqr_picp_list) if len(svqr_picp_list) > 0 else float('nan')
        svqr_mpiw = np.mean(svqr_mpiw_list) if len(svqr_mpiw_list) > 0 else float('nan')
        svqr_time = dataset_results['SVQR+CP'].get('time', float('nan'))
        
        summary_lines.append(f"SVQR+CP - PICP: {svqr_picp:.4f}, MPIW: {svqr_mpiw:.4f}, Time: {svqr_time:.2f}s")
        
        # CQR-NN results
        cqr_picp_all_runs = np.array(dataset_results['CQR-NN']['picp'])
        cqr_picp_mean = np.mean(cqr_picp_all_runs) if len(cqr_picp_all_runs) > 0 else float('nan')
        cqr_picp_std = np.std(cqr_picp_all_runs) if len(cqr_picp_all_runs) > 0 else float('nan')
        cqr_picp_min = np.min(cqr_picp_all_runs) if len(cqr_picp_all_runs) > 0 else float('nan')
        
        cqr_mpiw_all_runs = np.array(dataset_results['CQR-NN']['mpiw'])
        cqr_mpiw_mean = np.mean(cqr_mpiw_all_runs) if len(cqr_mpiw_all_runs) > 0 else float('nan')
        cqr_mpiw_std = np.std(cqr_mpiw_all_runs) if len(cqr_mpiw_all_runs) > 0 else float('nan')
        cqr_time_avg = dataset_results['CQR-NN'].get('time_avg', float('nan'))
        cqr_time_std = dataset_results['CQR-NN'].get('time_std', float('nan'))

        summary_lines.append(f"CQR-NN  - PICP: {cqr_picp_mean:.4f} (±{cqr_picp_std:.4f}) [Min: {cqr_picp_min:.4f}], " \
              f"MPIW: {cqr_mpiw_mean:.4f} (±{cqr_mpiw_std:.4f}), Time: {cqr_time_avg:.2f}s (±{cqr_time_std:.2f}s)")
        
        # Coverage deviation
        nominal_coverage = 0.9 # Assuming 90% nominal coverage
        svqr_dev = svqr_picp - nominal_coverage
        cqr_dev = cqr_picp_mean - nominal_coverage
        
        summary_lines.append(f"Coverage Deviation from {nominal_coverage*100}% nominal:")
        summary_lines.append(f"  SVQR+CP: {svqr_dev:+.4f}")
        summary_lines.append(f"  CQR-NN:  {cqr_dev:+.4f} (±{cqr_picp_std:.4f})")

    final_summary_text = "\n".join(summary_lines)
    print(final_summary_text)

    # Optionally, save this summary to a markdown file
    # md_results_path = os.path.join('../../results_new', 'experimental_summary.md')
    # with open(md_results_path, 'w') as f:
    #     f.write(final_summary_text.replace("=", "#").replace("-", "##")) # Basic md conversion
    # print(f"\nSummary saved to {md_results_path}")


if __name__ == "__main__":
    # Suppress warnings
    warnings.filterwarnings("ignore")
    
    # Run all experiments
    results_data = run_all_experiments(n_runs=5) # n_runs can be adjusted

