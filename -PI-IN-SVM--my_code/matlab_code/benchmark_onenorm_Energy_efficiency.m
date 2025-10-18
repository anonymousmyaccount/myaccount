
clear all;
close all;

% Load the dataset from an Excel file
filename = 'energy_efficiency.xlsx'; % Replace with your Parkinson's dataset file name
if ~isfile(filename)
    error('The specified file does not exist.');
end
data = readtable(filename);

% Check if data is non-empty
if isempty(data)
    error('The dataset is empty.');
end

% Extract predictor variables (X) and response variable (Y)
X = data{:, 1:end-2};  % Predictor variables
Y = data{:, end-1};      % Response variable

% Parameters for epsilon quantile SVR
s = 2^60;              % Kernel parameter
c1 = 2^-3.8;           % Regularization parameter
c3 = 0.1;              % Regularization parameter specific to One-Norm
lower_quantiles = [0.025, 0.050, 0.075];

% Kernel parameters
kerfPara.type = 'rbf';
kerfPara.pars = s;

% Initialize results cell array for storing results
results1 = cell(length(lower_quantiles) + 1, 7); % 7 columns
results1(1,:) = {'Tau', 'RMSE (Lower, Upper)', 'Time', 'Sparsity (Lower, Upper)', 'Coverage Probability (Lower, Upper)', 'PICP', 'MPIW'};

% Function to evaluate PICP and MPIW
function [PICP, MPIW] = evaluate_PICP(y, Low_Q, Up_Q)
    PICP = mean(y >= Low_Q & y <= Up_Q);
    MPIW = mean(Up_Q - Low_Q);
end

% Number of iterations
num_iterations = 25;

% Initialize arrays to store metrics across iterations
P_tau = zeros(num_iterations, length(lower_quantiles));
RMSE = zeros(num_iterations, length(lower_quantiles));
spars = zeros(num_iterations, length(lower_quantiles));
CP_lower = zeros(num_iterations, length(lower_quantiles));
CP_upper = zeros(num_iterations, length(lower_quantiles));
PICP = zeros(num_iterations, length(lower_quantiles));
MPIW = zeros(num_iterations, length(lower_quantiles));

% Split data into training (80%) and testing (20%) sets
cv = cvpartition(size(X, 1), 'Holdout', 0.5);
trainIdx = training(cv);
testIdx = test(cv);

trainX = X(trainIdx, :);
ytrain = Y(trainIdx, :);
testX = X(testIdx, :);
ytest = Y(testIdx, :);

% Iterate over each quantile
for i = 1:length(lower_quantiles)
    tau_lower = lower_quantiles(i);
    tau_upper = 0.90 + tau_lower;  % Complement of lower quantile

    % Loop for random seeds (rng 1:100)
    for j = 1:num_iterations
        rng(j); % Seed the random number generator for reproducibility

        % Predict the lower and upper quantiles using quantileLPONENORMTSVR
        tic; % Start timer
        [~, Low_Q, sparsity_lower] = quantileLPONENORMTSVR12(trainX, ytrain, testX, s, c3, c1, tau_lower);
        [~, Up_Q, sparsity_upper] = quantileLPONENORMTSVR12(trainX, ytrain, testX, s, c3, c1, tau_upper);
        elapsed_time = toc; % End timer

        % Calculate coverage probabilities for lower and upper quantiles
        CP_lower(j, i) = mean(ytest <= Low_Q);
        CP_upper(j, i) = mean(ytest <= Up_Q);

        % Evaluate PICP and MPIW using the evaluate_PICP function
        [PICP(j, i), MPIW(j, i)] = evaluate_PICP(ytest, Low_Q, Up_Q);

        % Calculate RMSE for lower and upper quantiles
        RMSE(j, i) = sqrt(mean((ytest - Low_Q).^2)); % RMSE for lower quantile
        RMSE(j, i + length(lower_quantiles)) = sqrt(mean((ytest - Up_Q).^2)); % RMSE for upper quantile

        % Store sparsity for lower and upper quantiles
        spars(j, i) = sparsity_lower;
        spars(j, i + length(lower_quantiles)) = sparsity_upper;
    end

    % Store average results across all iterations in the results cell array
    results1{i + 1, 1} = ['(', num2str(tau_lower), ', ', num2str(tau_upper), ')'];
    results1{i + 1, 2} = sprintf('%.4f, %.4f', mean(RMSE(:, i)), mean(RMSE(:, i + length(lower_quantiles)))); % RMSE
    results1{i + 1, 3} = sprintf('%.4f', mean(elapsed_time)); % Time
    results1{i + 1, 4} = sprintf('%.4f, %.4f', mean(spars(:, i)), mean(spars(:, i + length(lower_quantiles)))); % Sparsity
    results1{i + 1, 5} = sprintf('%.4f, %.4f', mean(CP_lower(:, i)), mean(CP_upper(:, i))); % Coverage Probability
    results1{i + 1, 6} = sprintf('%.4f', mean(PICP(:, i))); % PICP
    results1{i + 1, 7} = sprintf('%.4f', mean(MPIW(:, i))); % MPIW
end

% Display results
disp('Results:');
disp(results1);

% Save to CSV file
writetable(cell2table(results1), 'results_PI_onenorm_quantileLPONENORM_energy_efficiency.csv');
