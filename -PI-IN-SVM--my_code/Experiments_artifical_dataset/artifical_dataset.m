clear;
close all;
clc;
% Initialization of parameters
epsilons = 0:0.05:1; % Epsilon values
taus = 0.10:0.05:0.80; % Quantile levels
s = 2^0; % Kernel parameter
c3 = 2^0; % Regularization parameter for epsilon-SVR
c1 = 2^5; % Regularization parameter
N = 1500; % Total number of samples
n = 1000; % Number of test samples
% Initialize result storage
results = table('Size', [0, 8], ...
'VariableTypes', {'double', 'double', 'string', 'double', 'double', 'double', 'double', 'double'}, ...
'VariableNames', {'Epsilon', 'Tau', 'Dataset', 'CoverageProbability', 'CoverageError', 'PinballLoss', 'Sparsity', 'Time'});
% Generate artificial data for Dataset 1
a = -4;
b = 4;
X = a + (b - a) .* rand(N, 1);
rng(1); % For reproducibility
noise = 0.6 * randn(N, 1);
Y = (ones(N, 1) - X + 2 * (X .^ 2)) .* exp(-0.5 * (X .^ 2)) + noise;
% Train-test split for Dataset 1
train1 = X(1:N-n, :);
ytrain1 = Y(1:N-n, :);
test1 = X(N-n+1:N, :);
ytest1 = Y(N-n+1:N, :);
% Generate artificial data for Dataset 2
rng(1); % For reproducibility
noise1 = chi2rnd(3, N, 1);
Y = (ones(N, 1) - X + 2 * (X .^ 2)) .* exp(-0.5 * (X .^ 2)) + noise1;
% Train-test split for Dataset 2
train2 = X(1:N-n, :);
ytrain2 = Y(1:N-n, :);
test2 = X(N-n+1:N, :);
ytest2 = Y(N-n+1:N, :);
% Loop over each epsilon value
for epsilonIdx = 1:1
epsilon = epsilons(epsilonIdx);
% Loop over each tau value
for tauIdx = 1:length(taus)
tau = taus(tauIdx);
% Dataset 1
disp(['Processing Dataset 1 with epsilon = ', num2str(epsilon), ' and tau = ', num2str(tau)]);
% Start timing
tic;
[~, Ypredict1, sparsity1] = quantileLPONENORMTSVR1(train1, ytrain1, test1, s, c3, c1, tau);
% Stop timing
time1 = toc;
% Coverage Probability Calculation
CP1 = length(find(ytest1 < Ypredict1)) / n;
C_error1 = abs(CP1 - tau);
% Pinball Loss
error1 = pinloss(ytest1, Ypredict1, tau);
% Store results for Dataset 1
results = [results; {epsilon, tau, 'Dataset 1', CP1, C_error1, error1, sparsity1, time1}];
% Dataset 2
disp(['Processing Dataset 2 with epsilon = ', num2str(epsilon), ' and tau = ', num2str(tau)]);
% Start timing
tic;
[~, Ypredict2, sparsity2] = quantileLPONENORMTSVR1(train2, ytrain2, test2, s, c3, c1, tau);
% Stop timing
time2 = toc;
% Coverage Probability Calculation
CP2 = length(find(ytest2 < Ypredict2)) / n;
C_error2 = abs(CP2 - tau);
% Pinball Loss
error2 = pinloss(ytest2, Ypredict2, tau);
% Store results for Dataset 2
results = [results; {epsilon, tau, 'Dataset 2', CP2, C_error2, error2, sparsity2, time2}];
end
end
% Save results to CSV file
writetable(results, 'result_one_norm.csv');
% Function to calculate Pinball Loss
function loss = pinloss(ytrue, ypred, tau)
loss = mean(max((tau - (ytrue < ypred)) .* (ytrue - ypred), (tau - 1 + (ytrue < ypred)) .* (ypred - ytrue)));
end