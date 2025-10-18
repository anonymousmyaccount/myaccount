clear all;
close all;

% Load the dataset from a CSV file
filename = 'winequality-white.csv'; % Replace with your CSV file name
data = readtable(filename);

% Assuming the last column is the response variable (Y) and the rest are predictors (X)
X = data{:, 1:end-1};  % Extract predictor variables
Y = data{:, end};      % Extract response variable

% Parameters for epsilon quantile SVR
s = 2^6.9;              % Kernel parameter
c1 = 2^0.8;             % Regularization parameter
eps1 = 0;               % Epsilon value for SVR
lower_quantiles = [0.010, 0.015, 0.020, 0.025, 0.030];

% Train-test split (70% training, 30% testing)
N = size(X, 1);
n = round(0.3 * N);  % Number of test points
trainX = X(1:N-n,:);
ytrain = Y(1:N-n,:);
testX = X(N-n+1:end,:);
ytest = Y(N-n+1:end,:);

% Kernel parameters
kerfPara.type = 'rbf';
kerfPara.pars = s;

% Initialize results cell array for storing results
results1 = cell(length(lower_quantiles) + 1, 7); % 7 columns
results1(1,:) = {'Tau', 'RMSE (Lower, Upper)', 'Time', 'Sparsity (Lower, Upper)', 'Coverage Probability (Lower, Upper)', 'PICP', 'MPIW'};

% Iterate over each quantile
for i = 1:length(lower_quantiles)
    tau_lower = lower_quantiles(i);
    tau_upper = 0.95 + tau_lower;

    % Predict the lower and upper quantiles
    tic; % Start timer
    [Low_Q, ~,~,sparsity_lower] = epsilon_quantilesvr2(trainX, ytrain, testX, kerfPara, c1, tau_lower, eps1);
    [Up_Q, ~,~,sparsity_upper] = epsilon_quantilesvr2(trainX, ytrain, testX, kerfPara, c1, tau_upper, eps1);
    elapsed_time = toc; % End timer

    % Calculate coverage probabilities for lower and upper quantiles
    CP_lower = length(find(ytest <= Low_Q)) / n;
    CP_upper = length(find(ytest <= Up_Q)) / n;

    % Evaluate PICP and MPIW using the EVal_PI function
    [PICP, MPIW] = EVal_PI(ytest, Low_Q, Up_Q);

    % Calculate RMSE for lower and upper quantiles
    rmse_low = sqrt(mean((ytest - Low_Q).^2));
    rmse_up = sqrt(mean((ytest - Up_Q).^2));

    % Store results in the results cell array
    results1{i+1, 1} = ['(', num2str(tau_lower), ', ', num2str(tau_upper), ')'];
    results1{i+1, 2} = sprintf('%.4f, %.4f', rmse_low, rmse_up); % RMSE
    results1{i+1, 3} = sprintf('%.4f', elapsed_time); % Time
    results1{i+1, 4} = sprintf('%.4f, %.4f', sparsity_lower, sparsity_upper); % Sparsity
    results1{i+1, 5} = sprintf('%.4f, %.4f', CP_lower, CP_upper); % Coverage Probability
    results1{i+1, 6} = sprintf('%.4f', PICP); % PICP
    results1{i+1, 7} = sprintf('%.4f', MPIW); % MPIW
end

% Display results
disp('Results:');
disp(results1);

% Save to CSV file
writetable(cell2table(results1), 'result_PI_winequality-white_epsilon.csv');
