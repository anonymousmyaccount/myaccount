clear all;
close all;
tic
y = xlsread("beer.csv");
% plot(y,'b-');

%% Reduced Parameter Setup for Faster Execution
% Instead of a very fine grid, we choose fewer values.
win_size = [5, 10, 15, 20, 25,30,45,50];      % Only 5 window sizes
s1val   = -10:2:25;                 % 11 values for kernel exponent (coarser grid)
c1val   = -10:2:25;                 % 11 values for regularization constant

% Other fixed parameters
kernel    = 2;        % (not used directly)
q_lower   = 0.025;
q_upper   = 0.975;
target_cov= 0.95;     % target coverage threshold
c3        = 0.5;      % parameter for quantileLPONENORMTSVR12

% Preallocate arrays for evaluation metrics.
% Dimensions: (# win_size) x (# s1val) x (# c1val)
PICP_all = zeros(length(win_size), length(s1val), length(c1val));
MPIW_all = zeros(length(win_size), length(s1val), length(c1val));

%% Grid Search with Parallelization Over win_size
parfor w = 1:length(win_size)
    win = win_size(w);
    
    % Build sliding-window dataset using current win
    N = length(y);
    num_samples = N - win;
    X_all = zeros(num_samples, win);
    y_all = zeros(num_samples, 1);
    for i = 1:num_samples
        idx_window = i:(i+win-1);
        X_all(i,:) = y(idx_window, :);
        y_all(i,:) = y(idx_window(end) + 1, :);
    end
    
    % Split complete data into training (70%) and testing (30%)
    splitIdx = floor(num_samples * 0.7);
    X_train = X_all(1:splitIdx, :);
    y_train = y_all(1:splitIdx, :);
    X_test  = X_all(splitIdx+1:end, :);
    y_test  = y_all(splitIdx+1:end, :);
    
    % Local arrays for this iteration
    local_PICP = zeros(1, length(s1val), length(c1val));
    local_MPIW = zeros(1, length(s1val), length(c1val));
    
    % Loop over kernel parameter (s1) and regularization parameter (c1)
    for s = 1:length(s1val)
        s1_current = 2^(s1val(s));  % actual kernel parameter value
        s2_current = s1_current;    % using same value for both predictions
        % Create kernel parameter structs
        kerfPara1 = struct('type', 'rbf', 'pars', s1_current);
        kerfPara2 = struct('type', 'rbf', 'pars', s2_current);
        
        for c = 1:length(c1val)
            C1_current = 2^(c1val(c));  % actual regularization constant
            % Use the quantileLPONENORMTSVR12 function to predict the lower and upper quantiles
            [Low_Q, Pred_test_lower, sparsity_lower] = quantileLPONENORMTSVR12(X_train, y_train, X_test, s1_current, c3, C1_current, q_lower);
            [Up_Q, Pred_test_upper, sparsity_upper]   = quantileLPONENORMTSVR12(X_train, y_train, X_test, s2_current, c3, C1_current, q_upper);
            
            % Evaluate Prediction Interval Coverage Probability (PICP) and 
            % Mean Prediction Interval Width (MPIW) on the test set.
            [PICP_val, MPIW_val] = evaluate_PICP(y_test, Pred_test_lower, Pred_test_upper);
            local_PICP(1, s, c) = PICP_val;
            local_MPIW(1, s, c) = MPIW_val;
        end
    end
    % Store local results into global arrays.
    PICP_all(w, :, :) = local_PICP;
    MPIW_all(w, :, :) = local_MPIW;
end
toc

%% (Optional) Display one example result from the grid search.
% For demonstration, choose parameters: win = win_size(1), s1 = s1val(6), c1 = c1val(6)
win_chosen = win_size(1);
s1_chosen = s1val(6);  % adjust index as needed
c1_chosen = c1val(6);

% Rebuild the dataset using the chosen window size
N = length(y);
num_samples = N - win_chosen;
X_all = zeros(num_samples, win_chosen);
y_all = zeros(num_samples, 1);
for i = 1:num_samples
    idx_window = i:(i+win_chosen-1);
    X_all(i,:) = y(idx_window, :);
    y_all(i,:) = y(idx_window(end) + 1, :);
end
splitIdx = floor(num_samples * 0.7);
X_train = X_all(1:splitIdx, :);
y_train = y_all(1:splitIdx, :);
X_test  = X_all(splitIdx+1:end, :);
y_test  = y_all(splitIdx+1:end, :);

% Set the chosen parameters
kerfPara1 = struct('type', 'rbf', 'pars', 2^(s1_chosen));
kerfPara2 = struct('type', 'rbf', 'pars', 2^(s1_chosen));
C1_chosen = 2^(c1_chosen);

% Compute predictions for the test set using chosen parameters.
[Low_Q, Pred_test_lower, sparsity_lower] = quantileLPONENORMTSVR12(X_train, y_train, X_test, 2^(s1_chosen), c3, C1_chosen, q_lower);
[Up_Q, Pred_test_upper, sparsity_upper]   = quantileLPONENORMTSVR12(X_train, y_train, X_test, 2^(s1_chosen), c3, C1_chosen, q_upper);

% Evaluate and display final metrics.
[PICP_final, MPIW_final] = evaluate_PICP(y_test, Pred_test_lower, Pred_test_upper);
fprintf('Final Evaluation: PICP = %f, MPIW = %f\n', PICP_final, MPIW_final);

% Plot the test set predictions.
figure;
hold on;
plot(y_test, 'b-', 'DisplayName', 'True Values');
plot(Pred_test_lower, 'r-', 'DisplayName', 'Predicted Lower Quantile');
plot(Pred_test_upper, 'k-', 'DisplayName', 'Predicted Upper Quantile');
legend;
title('Prediction Intervals on Test Data');
hold off;

%% Helper Function: Evaluate PICP and MPIW
function [PICP, MPIW] = evaluate_PICP(y, Low_Q, Up_Q)
    PICP = mean(y >= Low_Q & y <= Up_Q);
    MPIW = mean(Up_Q - Low_Q);
end
