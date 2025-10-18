clear all;
close all;
tic
y = xlsread("Daily-total-female-births.csv");
% plot(y,'b-');

%% Parameter Setup
win_size = 1:25;       % window sizes to try
s1val = -25:1:25;      % exponents for kernel parameter
c1val = -25:1:25;      % exponents for regularization constant

% Other fixed parameters
kernel = 2;            % (not used directly here)
q_lower = 0.025;
q_upper = 0.975;
target_cov = 0.95;     % (for reference)
c3 = 0.5;              % parameter for quantileLPONENORMTSVR12

% Preallocate arrays for evaluation metrics.
% Dimensions: (# win_size) x (# s1val) x (# c1val)
PICP_all = zeros(length(win_size), length(s1val), length(c1val));
MPIW_all = zeros(length(win_size), length(s1val), length(c1val));

%% Grid Search with Parallelization on win_size
parfor w = 1:length(win_size)
    win = win_size(w);
    
    % Build sliding-window dataset using current win
    X_all = [];
    y_all = [];
    for i = 1:(length(y) - win)
        idx_window = i:(i + win - 1);
        X_all(i, :) = y(idx_window, :);
        y_all(i, :) = y(idx_window(end) + 1, :);
    end
    
    % Split complete data into training (70%) and test (30%)
    n_total = size(X_all, 1);
    idx_split = floor(n_total * 0.7);
    X_train = X_all(1:idx_split, :);
    y_train = y_all(1:idx_split, :);
    X_test  = X_all(idx_split+1:end, :);
    y_test  = y_all(idx_split+1:end, :);
    
    % Local arrays for results in this parfor iteration
    local_PICP = zeros(1, length(s1val), length(c1val));
    local_MPIW = zeros(1, length(s1val), length(c1val));
    
    % Loop over kernel parameter (s1) and regularization parameter (c1)
    for s = 1:length(s1val)
        % For both lower and upper predictions use the same kernel value.
        s1_current = s1val(s);   % exponent value; actual parameter = 2^(s1_current)
        s2_current = s1_current;  % same for s2
        % Create the kernel parameter structs in one step.
        kerfPara1 = struct('type','rbf','pars',2^s1_current);
        kerfPara2 = struct('type','rbf','pars',2^s2_current);
        
        for c = 1:length(c1val)
            % Set regularization constant (C1 and C2 are set to the same value here)
            C1_current = 2^(c1val(c));
            % Predict using the quantileLPONENORMTSVR12 function for the test set.
            [Low_Q, Pred_test_lower, sparsity_lower] = quantileLPONENORMTSVR12(X_train, y_train, X_test, 2^s1_current, c3, C1_current, q_lower);
            [Up_Q, Pred_test_upper, sparsity_upper]   = quantileLPONENORMTSVR12(X_train, y_train, X_test, 2^s2_current, c3, C1_current, q_upper);
            
            % Evaluate Prediction Interval Coverage Probability (PICP) and 
            % Mean Prediction Interval Width (MPIW) on the test set.
            [PICP_val, MPIW_val] = evaluate_PICP(y_test, Pred_test_lower, Pred_test_upper);
            local_PICP(1, s, c) = PICP_val;
            local_MPIW(1, s, c) = MPIW_val;
        end
    end
    % Store the local results into the global arrays.
    PICP_all(w, :, :) = local_PICP;
    MPIW_all(w, :, :) = local_MPIW;
end
toc

%% (Optional) Display one example result from the grid search
% For demonstration, choose parameters: win = win_size(1), s1 = s1val(18), c1 = c1val(31)
win_chosen = 3  %win_size(1);
s1_chosen = 0.0039*2 %
c1_chosen = c1val(31) %   2^31; %c1val(31);
c3 = 32;


% Rebuild the dataset using the chosen win
X_all = [];
y_all = [];
for i = 1:(length(y) - win_chosen)
    idx_window = i:(i + win_chosen - 1);
    X_all(i, :) = y(idx_window, :);
    y_all(i, :) = y(idx_window(end) + 1, :);
end
n_total = size(X_all, 1);
idx_split = floor(n_total * 0.7);
X_train = X_all(1:idx_split, :);
y_train = y_all(1:idx_split, :);
X_test  = X_all(idx_split+1:end, :);
y_test  = y_all(idx_split+1:end, :);

% Set kernel and regularization parameters for the chosen combination.
kerfPara1 = struct('type','rbf','pars',2^s1_chosen);
kerfPara2 = struct('type','rbf','pars',2^s1_chosen);
C1_chosen = 2^(c1_chosen);

% Get predictions for the test set.
[Low_Q, Pred_test_lower, sparsity_lower] = quantileLPONENORMTSVR12(X_train, y_train, X_test, 2^s1_chosen, c3, C1_chosen, q_lower);
[Up_Q, Pred_test_upper, sparsity_upper]   = quantileLPONENORMTSVR12(X_train, y_train, X_test, 2^s1_chosen, c3, C1_chosen, q_upper);

% Evaluate and display the final metrics.
[PICP_final, MPIW_final] = evaluate_PICP(y_test, Pred_test_lower, Pred_test_upper);
fprintf('Final Evaluation for chosen parameters: PICP = %f, MPIW = %f\n', PICP_final, MPIW_final);

% Plot the test set predictions.
figure;
hold on;
plot(y_test, 'b-', 'DisplayName',
', 'True Values');
plot(Pred_test_lower, 'r-', 'DisplayName', 'Predicted Lower Quantile');
plot(Pred_test_upper, 'k-', 'DisplayName', 'Predicted Upper Quantile');
legend;
title('Prediction Intervals on Test Data');
hold off;

%% Helper Function to Evaluate PICP and MPIW
function [PICP, MPIW] = evaluate_PICP(y, Low_Q, Up_Q)
    PICP = mean(y >= Low_Q & y <= Up_Q);
    MPIW = mean(Up_Q - Low_Q);
end
