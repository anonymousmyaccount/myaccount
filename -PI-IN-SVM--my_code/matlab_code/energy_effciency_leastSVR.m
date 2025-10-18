% Step 1: Load your data from an Excel file
filename = 'ENB2012_data.xlsx';  % Replace with your actual file name
data = readtable(filename);  % Load the data

% Step 2: Extract features (X) and response (Y)
X = table2array(data(:, 1:end-2));  % All columns except the last one (features)
Y = table2array(data(:,end-1));      % Last column as the response variable

% Step 3: Normalize X using Z-score normalization
X_normalized= (X- mean(X))./ std(X)

% Step 4: Split the data into training and test sets

X_train = X_normalized(1:floor(size(X_normalized,1)*0.8),:);
Y_train =Y(1:floor(size(X_normalized,1)*0.8),:);
X_test =  X_normalized(floor(size(X_normalized,1)*0.8)+1:end,:);
Y_test = Y(floor(size(X_normalized,1)*0.8)+1:end,:);

% Step 6: Set kernel parameters, regularization, and epsilon values for SVR
kerfPara.type = 'rbf';  % Radial Basis Function (RBF) kernel
kerfPara.pars = 2^2.5;      % Kernel parameter (gamma for RBF)
C = 2^5;              % SVR with tau = 0.5 for standard regression
eps1 = 0;              % Epsilon-insensitive margin

% Step 7: Train the SVR model and make predictions using your custom function
%[PredictY, f1, nsv, sparsity] = epsilon_quantilesvr2(X_train_normalized, Y_train, X_test_normalized, kerfPara, C, tau, eps1);
[PredictY, f1,sparsity] = leastsquaresvr(X_train, Y_train, X_test, kerfPara, C);
                
% Step 8: Evaluate the results (Compare predictions with actual test values)
mseValue = mean((PredictY - Y_test).^2);
fprintf('Mean Squared Error: %.4f\n', mseValue);


% Step 9: Calculate and print the error for each predicted value
errors = Y_test - PredictY;  % The difference between predicted and actual values
fprintf('Index\tPredicted Value\tActual Value\tError\n');
for i = 1:length(Y_test)
    fprintf('%d\t%.4f\t\t%.4f\t\t%.4f\n', i, PredictY(i), Y_test(i), errors(i));
end

% Step 10: Save errors to an Excel file
filename1 = 'prediction_errors.xlsx';  % Define the filename for Excel
dataToSave = table((1:length(Y_test))', Y_test, PredictY, errors, ...
                   'VariableNames', {'Index', 'ActualValue', 'PredictedValue', 'Error'});
writetable(dataToSave, filename1);  % Save the table to an Excel file

% Step 11: Plot the errors
figure;
plot(errors, '-o');  % Plot the error values
xlabel('Test Data Index');
ylabel('Prediction Error');
title('Prediction Errors for ',filename);
grid on;

% Step 12: Calculate the standard deviation of the errors
std_dev_errors = std(errors);
fprintf('Standard Deviation of Errors: %.4f\n', std_dev_errors);

% Step 13: Define the desired quantiles and calculate them
mu = mean(errors);
sigma = std(errors);
% Define quantile pairs
quantiles_to_compute_pairs = [0.025, 0.925;0.05, 0.95;0.075, 0.975];

% Initialize arrays to store results
taus = [];      % Tau values for each quantile pair
picp_values = []; % Prediction Interval Coverage Probability (PICP)
mpiw_values = []; % Mean Prediction Interval Width (MPIW)
sparsity_values = []; % Sparsity values

% Loop over each pair of quantiles
for j = 1:size(quantiles_to_compute_pairs, 1)
  
    % Extract the current quantile pair
    lower_quantile = quantiles_to_compute_pairs(j, 1);
    upper_quantile = quantiles_to_compute_pairs(j, 2);
    taus = [taus; lower_quantile, upper_quantile];  % Store tau pair

    % Calculate normal quantiles based on sigma (assuming sigma is defined)
    quantile_values_normal = norminv([lower_quantile, upper_quantile], 0, sigma);

    % Calculate prediction intervals based on quantiles
    lower_bound = PredictY + quantile_values_normal(1);  % Lower bound
    upper_bound = PredictY + quantile_values_normal(2);  % Upper bound

    % Calculate PICP and MPIW for current quantile pair
    [PICP, MPIW] = EVal_PI(Y_test, lower_bound, upper_bound);
    % fprintf('hi %.4f',PICP)

    % Calculate sparsity (assuming a sparsity calculation function is defined)
      % Replace with actual sparsity calculation
    sparsity_values = [sparsity; sparsity];

    % Store PICP and MPIW values for the current pair
    picp_values = [picp_values; PICP];
    mpiw_values = [mpiw_values; MPIW];

    rmse_low = sqrt(mean((Y_test - lower_quantile ).^2));
    rmse_up = sqrt(mean((Y_test  - upper_quantile ).^2));

figure;
hold on;
% Plot actual test values
plot(1:length(Y_test), Y_test, 'o-', 'DisplayName', 'Actual Values', 'MarkerFaceColor', 'blue');
% Plot predicted values
plot(1:length(PredictY), PredictY, 'x-', 'DisplayName', 'Predicted Values', 'MarkerFaceColor', 'red');
% Plot lower and upper bounds
plot(1:length(lower_bound), lower_bound, '--', 'DisplayName', 'Lower Bound', 'Color', [0.2, 0.7, 0.2]);
plot(1:length(upper_bound), upper_bound, '--', 'DisplayName', 'Upper Bound ', 'Color', [0.8, 0.2, 0.2]);
% Add plot details
xlabel('Test Data Index');
ylabel('Value');
title('Actual vs Predicted Values with Prediction Intervals for interval for interval  ',[lower_quantile, upper_quantile]);
legend;
grid on;
hold off;

end

% % Create a table to store results for each tau pair
% results_table = table(taus, picp_values, mpiw_values, sparsity_values, ...
%     {'Tau_Pairs', 'PICP', 'MPIW', 'Sparsity'});
% 
% % Display table
% disp(results_table);

% Save table to an Excel file
% filename = 'Quantile_Results_Table.xlsx';
% writetable(results_table, filename);
disp(length(taus));
disp(length(picp_values));
disp(length(mpiw_values));
disp(length(sparsity_values));
% Print results for each tau pair
fprintf('The results for %s \n',filename)
fprintf('Tau Pair\tPICP\tMPIW\n');
for i = 1:size(taus, 1)
    fprintf('(%.3f, %.3f)\t%.4f\t%.4f\n', taus(i, 1), taus(i, 2), picp_values(i), mpiw_values(i));
end


