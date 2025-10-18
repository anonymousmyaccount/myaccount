% Load the dataset and preserve original column headers
data = readtable('san francisco.csv', 'VariableNamingRule', 'preserve');
% Convert the 'Date/Time' column to datetime format if it's not already
data.('Date/Time') = datetime(data.('Date/Time'), 'InputFormat', 'yyyy-MM-dd HH:mm:ss');
% Rename the 'Date/Time' column to your desired name (e.g., 'DateTime')
data.Properties.VariableNames{'Date/Time'} = 'DateTime';
% Display the updated column names for verification
disp('Updated Column Names:');
disp(data.Properties.VariableNames);
% Plot wind speeds (assuming wind speeds are in columns 2 to end)
figure;
plot(data.DateTime, data{:, 3});  % Plot all wind speeds (columns 2-5)
title('Wind Speed Over Time');
xlabel('Datetime');
ylabel('Wind Speed (m/s)');
legend(data.Properties.VariableNames(:,2));  % Use column names for legend
grid on;
% Save the plot as an image
saveas(gcf, 'jaisalmer.png');
% Optional: Split the dataset into training, validation, and test sets
input_window_size = 24;  % Define the window size
validation_size = 0.15;  % Validation size (15%)
test_size = 0.15;        % Test size (15%)
% Split indices for validation and test sets
split_index_val = int64(height(data) * (1 - validation_size - test_size));
split_index_test = int64(height(data) * (1 - test_size));
% Create training, validation, and test datasets
train_data = data(1:split_index_val, :);
validation_data = data(split_index_val + 1:split_index_test, :);
test_data = data(split_index_test + 1:end, :);
% Truncate function to create input-output pairs
function [in_, out_] = truncate(x, feature_cols, target_cols, train_len, test_len)
    in_ = []; out_ = [];
    for i = 1:(length(x) - train_len - test_len + 1)
        in_ = [in_; x(i:(i + train_len - 1), feature_cols)'];
        out_ = [out_; x((i + train_len):(i + train_len + test_len - 1), target_cols)'];
    end
    in_ = in_';  % Transpose to get the correct dimensions
    out_ = out_';  % Transpose to get the correct dimensions
end
% Prepare training, validation, and test input-output pairs
[train_x, train_y] = truncate(train_data{:, 2:end}, 1:size(train_data, 2)-1, 1, input_window_size, 1);
[val_x, val_y] = truncate(validation_data{:, 2:end}, 1:size(validation_data, 2)-1, 1, input_window_size, 1);
[test_x, test_y] = truncate(test_data{:, 2:end}, 1:size(test_data, 2)-1, 1, input_window_size, 1);
% Display the sizes of the created datasets
disp('Training set size:');
disp(size(train_x));
disp(size(train_y));
disp('Validation set size:');
disp(size(val_x));
disp(size(val_y));
disp('Test set size:');
disp(size(test_x));
disp(size(test_y));
% Parameters for Epsilon SVR
b = 4; 
a = -4;
s = 2^0;               % Kernel parameter
c1 = 2^0;              % Regularization parameter
c3 = 0;
eps1 = 0;            % Epsilon value for SVR
N = 1500;               % Total samples in dataset
n = 1000;              % Number of test samples
lower_quantiles = [0.010, 0.015, 0.020, 0.025, 0.030];
% Initialize results cell array for Artificial Dataset
results1 = cell(length(lower_quantiles)+1, 7); % 7 columns
% Headers
results1(1,:) = {'Tau', 'RMSE (Lower, Upper)', 'Time', 'Sparsity (Lower, Upper)', 'Coverage Probability (Lower, Upper)', 'PICP', 'MPIW'};
% Train-test split from the wind speed dataset
% Assuming you have already defined X and Y based on your dataset
X = train_x;  % Use your training data for X
ytrain = train_y; % Use your training data for Y
testX = test_x; % Test input
ytest = test_y; % Test output
kerfPara.type = 'rbf';
kerfPara.pars = s;
% Iterate over each quantile
for i = 1:length(lower_quantiles)
    tau_lower = lower_quantiles(i);
    tau_upper = 0.95 + tau_lower; 
    % Predict the lower and upper quantiles
    tic; % Start timer
    %[~, Low_Q, sparsity_lower] = quantileLPONENORMTSVR12(trainX, ytrain, testX, s, c3, c1, tau_lower);
    %98[~, Up_Q, sparsity_upper] = quantileLPONENORMTSVR12(trainX, ytrain, testX, s, c3, c1, tau_upper);
    [~,Low_Q,sparsity_lower] =  quantileLPONENORMTSVR12(X', ytrain', testX, s, c3, c1, tau_lower);
    [~, Up_Q,sparsity_upper] = quantileLPONENORMTSVR12(X, ytrain, testX, s, c3, c1, tau_upper);
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
disp('Results for Wind Speed Dataset:');
disp(results1);
% Save to CSV file
writecell(results1, ['results_PI_San_Fransico_Wind_Speed_Dataset_OneNorm.csv']);