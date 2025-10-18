% Define parameter ranges for s and c1 (logarithmic scale)
s_range = 2.^(-5:1:20);
c1_range = 2.^(-5:1:20);

best_s = NaN;
best_c1 = NaN;
best_PICP = 0;
best_results = [];

% Initialize variables for grid search
for s_idx = 1:length(s_range)
    for c1_idx = 1:length(c1_range)
        s = s_range(s_idx);
        c1 = c1_range(c1_idx);
        
        % Initialize arrays to store metrics across iterations
        PICP_avg = 0;
        MPIW_avg = 0;
        RMSE_avg = 0;

        % Perform cross-validation
        for fold = 1:5 % 5-fold cross-validation
            cv = cvpartition(size(X, 1), 'KFold', 5);
            trainIdx = training(cv, fold);
            testIdx = test(cv, fold);
            
            trainX = X(trainIdx, :);
            ytrain = Y(trainIdx, :);
            testX = X(testIdx, :);
            ytest = Y(testIdx, :);
            
            % Call your function to get predictions and sparsity
            [trainf1, f1, sparsity] = quantileLPONENORMTSVR12(trainX, ytrain, testX, s, c3, c1, tau_lower);
            [trainf1, f1_upper, sparsity_upper] = quantileLPONENORMTSVR12(trainX, ytrain, testX, s, c3, c1, tau_upper);

            % Evaluate PICP and MPIW for this fold
            [PICP, MPIW] = evaluate_PICP(ytest, f1, f1_upper);
            
            % Calculate RMSE for lower and upper quantiles
            RMSE_lower = sqrt(mean((ytest - f1).^2));
            RMSE_upper = sqrt(mean((ytest - f1_upper).^2));
            RMSE = (RMSE_lower + RMSE_upper) / 2;

            % Accumulate metrics
            PICP_avg = PICP_avg + PICP / 5;
            MPIW_avg = MPIW_avg + MPIW / 5;
            RMSE_avg = RMSE_avg + RMSE / 5;
        end
        
        % Update the best parameters based on PICP and RMSE
        if PICP_avg >= target_PICP && RMSE_avg < best_PICP
            best_s = s;
            best_c1 = c1;
            best_PICP = PICP_avg;
            best_results = [PICP_avg, MPIW_avg, RMSE_avg];
        end
    end
end

% Display best parameters and corresponding performance
fprintf('Best s: %.4f, Best c1: %.4f\n', best_s, best_c1);
fprintf('PICP: %.4f, MPIW: %.4f, RMSE: %.4f\n', best_results(1), best_results(2), best_results(3));
