%% quantileSVR.m
% MATLAB implementation of Epsilon-based QP Quantile Regression
% Converted from the provided Python code

clear; clc;

%% 1. Load and Prepare the Dataset
fprintf('Loading dataset ...\n');
% Read the CSV file. Adjust the filename/path as needed.
dataTable = readtable('Daily-total-female-births.csv');

% Ensure the table has the correct column names.
if ~any(strcmp(dataTable.Properties.VariableNames, 'date'))
    error('The dataset must contain a "DATE" column.');
end
if ~any(strcmp(dataTable.Properties.VariableNames, 'births'))
    % Assume the second column is Births if not already named.
    dataTable.Properties.VariableNames{2} = 'Births';
end

% Convert DATE column to datetime and sort the data.
dataTable.DATE = datetime(dataTable.date);
dataTable = sortrows(dataTable, 'date');
dataTable = rmmissing(dataTable, 'DataVariables', 'births');
dataTable.Births = double(dataTable.births);

% Create a time index normalized between 0 and 1.
nSamples = height(dataTable);
TimeIndex = linspace(0, 1, nSamples)';
dataTable.TimeIndex = TimeIndex;

% Prepare features and target.
X = TimeIndex;         % Feature: time index (n x 1)
y = dataTable.Births;  % Target variable
y_mean = mean(y);
y_std = std(y);
y_norm = (y - y_mean) / y_std;
fprintf('Dataset loaded with %d samples.\n\n', nSamples);

%% 2. Grid Search for Hyperparameters using epsilon_quantilesvr2
% Candidate hyperparameters.
window_size_candidates = [3];    % window size (number of days)
s_candidates = [2.0];               % kernel parameter (gamma in RBF)
C_candidates = [10.0];              % regularization parameter
eps1_candidates = [0.0];            % interpolation error parameter

q_lower = 0.025;
q_upper = 0.975;
step_size = 3;        % sliding window step size
target_cov = 0.95;    % target coverage for prediction intervals

results = [];  % each row: [win, s, C, eps1, PICP, MPIW, runtime]
total_candidates = numel(window_size_candidates) * numel(s_candidates) * ...
                   numel(C_candidates) * numel(eps1_candidates);
candidate_count = 0;

fprintf('Starting grid search over hyperparameters using epsilon_quantilesvr2 ...\n');






for win = window_size_candidates














for win = window_size_candidates
    for s_val = s_candidates
        for C_val = C_candidates
            for eps1_val = eps1_candidates
                candidate_count = candidate_count + 1;
                fprintf('\n[Grid Search] Processing candidate %d/%d: window=%d, s=%.2f, C=%.2f, eps1=%.2f\n',...
                    candidate_count, total_candidates, win, s_val, C_val, eps1_val);
                candidate_start = tic;
                temp_pred_lower = [];
                temp_pred_upper = [];
                temp_y = [];
                
                total_windows = floor((length(X) - win) / step_size);
                for i = 1:(length(X) - win)
                    idx_window = i:(i+win-1);
                    % Report progress every 100 iterations.
                    if mod((i-1)/step_size, 100) == 0
                        fprintf('  - Sliding window iteration %d/%d\n', floor((i-1)/step_size)+1, total_windows);
                    end
                    X_train(i,:) = y(idx_window, :);
                    y_train(i,:) =y(idx_window(end)+1,:);
                end

              Xtest= X_train(floor(length(y_train)*0.7)+1:end,:);      
              Ytest= y_train(floor(length(y_train)*0.7)+1:end,:);
              X_train= X_train(1:floor(length(y_train)*0.7),:); 
              y_train= y_train(1:floor(length(y_train)*0.7),:);
              Xval=  X_train(floor(length(y_train)*0.9)+1:end,:);
              Yval= y_train(floor(length(y_train)*0.9)+1:end,:);
              X_train= X_train(1:floor(length(y_train)*0.9),:); 
              y_train= y_train(1:floor(length(y_train)*0.9),:);


                    kerfPara.type = 'rbf';
                    kerfPara.pars = s_val;
                    try
                        % Lower quantile prediction.
                        [~, f_lower, ~, ~] = epsilon_quantilesvr2(X_train, y_train, Xval, kerfPara, C_val, q_lower, eps1_val);
                        % Upper quantile prediction.
                        [~, f_upper, ~, ~] = epsilon_quantilesvr2(X_train, y_train, Xval, kerfPara, C_val, q_upper, eps1_val);
                    catch ME
                        fprintf('    * Candidate window=%d, s=%.2f, C=%.2f, eps1=%.2f did not converge: %s\n',...
                            win, s_val, C_val, eps1_val, ME.message);
                        continue;
                    end
                    
                    temp_pred_lower(en = f_lower;
                    temp_pred_upper(end+1,1) = f_upper;
                    temp_y(end+1,1) = actual;
                end
                
                if isempty(temp_pred_lower)
                    fprintf('  * No predictions for candidate window=%d, s=%.2f, C=%.2f, eps1=%.2f\n',...
                        win, s_val, C_val, eps1_val);
                    continue;
                end
                
                % Rescale predictions to the original scale.
               
                
                candidate_picp = mean((Yval >= f_lower) & (Yval <= f_upper));
                candidate_mpiw = mean(temp_pred_upper_orig - temp_pred_lower_orig);
                candidate_runtime = toc(candidate_start);
                
                results = [results; win, s_val, C_val, eps1_val, candidate_picp, candidate_mpiw, candidate_runtime];
                
                fprintf('Candidate result: PICP=%.2f%%, MPIW=%.2f, RunTime=%.2fs\n',...
                    candidate_picp * 100, candidate_mpiw, candidate_runtime);
            end
        end
    end
end

% Select best candidate based on target coverage and minimum MPIW.
if isempty(results)
    error('No candidate converged in the grid search.');
end

candidates_meeting_target = results(results(:,5) >= target_cov, :);
if ~isempty(candidates_meeting_target)
    [~, best_idx] = min(candidates_meeting_target(:,6));  % minimum MPIW
    best_candidate = candidates_meeting_target(best_idx, :);
else
    % Otherwise choose candidate with maximum PICP and then minimum MPIW.
    sorted_results = sortrows(results, [-5, 6]);
    best_candidate = sorted_results(1, :);
end

fprintf('\nBest candidate parameters (epsilon_quantilesvr2): window=%d, s=%.2f, C=%.2f, eps1=%.2f, PICP=%.2f%%, MPIW=%.2f, RunTime=%.2fs\n',...
    best_candidate(1), best_candidate(2), best_candidate(3), best_candidate(4), ...
    best_candidate(5)*100, best_candidate(6), best_candidate(7));

window_size_final = best_candidate(1);
s_final = best_candidate(2);
C_final = best_candidate(3);
eps1_final = best_candidate(4);

%% 3. Final Forecasting using Best Hyperparameters
fprintf('\nStarting final forecasting using best hyperparameters ...\n');
pred_lower_list = [];
pred_upper_list = [];
actual_list = [];
dates_list = [];

total_forecast = floor((length(X) - window_size_final) / step_size);
for i = 1:step_size:(length(X) - window_size_final)
    if mod((i-1)/step_size, 100) == 0
        fprintf('  - Final forecasting iteration %d/%d\n', floor((i-1)/step_size)+1, total_forecast);
    end
    idx_window = i:(i+window_size_final-1);
    X_train = X(idx_window, :);
    y_train = y_norm(idx_window);
    test_index = i + window_size_final;
    X_test = X(test_index, :);
    actual = y_norm(test_index);
    
    kerfPara.type = 'rbf';
    kerfPara.pars = s_final;
    [~, f_lower, ~, ~] = epsilon_quantilesvr2(X_train, y_train, X_test, kerfPara, C_final, q_lower, eps1_final);
    [~, f_upper, ~, ~] = epsilon_quantilesvr2(X_train, y_train, X_test, kerfPara, C_final, q_upper, eps1_final);
    
    pred_lower_list(end+1,1) = f_lower;
    pred_upper_list(end+1,1) = f_upper;
    actual_list(end+1,1) = actual;
    dates_list(end+1,1) = dataTable.DATE(test_index);
end

pred_lower_arr = pred_lower_list;
pred_upper_arr = pred_upper_list;
actual_arr = actual_list;

% Rescale predictions to original scale.
pred_lower_orig = pred_lower_arr * y_std + y_mean;
pred_upper_orig = pred_upper_arr * y_std + y_mean;
actual_orig = actual_arr * y_std + y_mean;

picp = mean((actual_orig >= pred_lower_orig) & (actual_orig <= pred_upper_orig));
mpiw = mean(pred_upper_orig - pred_lower_orig);

fprintf('\nFinal Model Parameters (epsilon_quantilesvr2): window=%d, s=%.2f, C=%.2f, eps1=%.2f\n',...
    window_size_final, s_final, C_final, eps1_final);
fprintf('Final PICP: %.2f%%\n', picp * 100);
fprintf('Final MPIW: %.2f\n', mpiw);

%% 4. Plot the Results
figure('Position', [100 100 800 400]);
plot(dates_list, actual_orig, 'k-', 'DisplayName', 'True Values'); hold on;
plot(dates_list, pred_lower_orig, 'b--', 'DisplayName', 'Lower Quantile');
plot(dates_list, pred_upper_orig, 'r--', 'DisplayName', 'Upper Quantile');
fill([dates_list; flipud(dates_list)], [pred_lower_orig; flipud(pred_upper_orig)], 'g', ...
    'FaceAlpha', 0.2, 'EdgeColor', 'none');
datetick('x','yyyy-mm-dd','keepticks');
legend('show');
title('Daily Total Female Births Prediction with Epsilon Quantile Regression');
xlabel('Date');
ylabel('Daily Total Female Births');
xtickangle(45);
grid on;
hold off;



