clear all;
close all;
y = xlsread('beer.csv');
%plot(y,'b-');
%%
win =7;
kernel=2;               % window size (number of days)
s1= 2^50;
s2=2^20;                       
C1=2^25;
C2= 2^25;
q_lower = 0.025;
q_upper = 0.975;
target_cov = 0.95;
c3=0.25;
%%
   for i = 1:(length(y) - win)
                    idx_window = i:(i+win-1);
                    X_train(i,:) = y(idx_window, :);
                    y_train(i,:) =y(idx_window(end)+1,:);
   end

              Xtest= X_train(floor(length(y_train)*0.7)+1:end,:);      
              ytest= y_train(floor(length(y_train)*0.7)+1:end,:);
              X_train= X_train(1:floor(length(y_train)*0.7),:); 
              y_train= y_train(1:floor(length(y_train)*0.7),:);
              Xval=  X_train(floor(length(y_train)*0.9)+1:end,:);
              yval= y_train(floor(length(y_train)*0.9)+1:end,:);
              X_train= X_train(1:floor(length(y_train)*0.9),:); 
              y_train= y_train(1:floor(length(y_train)*0.9),:);
%%

    kerfPara1.type = 'rbf';
   kerfPara1.pars = s1;
    kerfPara2.type = 'rbf';
   kerfPara2.pars = s2;          
                      % Lower quantile prediction.
                      [Pred_val_lower, f_lower, ~, ~] = epsilon_quantilesvr2(X_train, y_train, Xval, kerfPara1, C1, q_lower, 0);
                        % Upper quantile prediction.
                      [Pred_val_upper, f_upper, ~, ~] = epsilon_quantilesvr2(X_train, y_train, Xval, kerfPara2, C2, q_upper,0);
                       %  [Pred_val_lower, f_lower, ~, ~] = epsilon_quantilesvr2(X_train, y_train, Xtest, kerfPara1, C1, q_lower, 0);
                        % Upper quantile prediction.
                      %[Pred_val_upper, f_upper, ~, ~] = epsilon_quantilesvr2(X_train, y_train, Xtest, kerfPara2, C2, q_upper,0);

%[Low_Q,Pred_val_lower, sparsity_lower] = quantileLPONENORMTSVR12(X_train, y_train, Xval, s1, c3, C1, q_lower);
%[Up_Q,Pred_val_upper,  sparsity_upper] = quantileLPONENORMTSVR12(X_train, y_train, Xval, s2, c3, C1, q_upper);
     %%                 
     target=yval;
    [PICP, MPIW] = evaluate_PICP(target,Pred_val_lower, Pred_val_upper);
function [PICP, MPIW] = evaluate_PICP(y, Low_Q, Up_Q)
    PICP = mean(y >= Low_Q & y <= Up_Q)
    MPIW = mean(Up_Q - Low_Q)
end
hold on;
plot(yval, ' b-');
hold on;
plot(Pred_val_lower, ' r-');
plot(Pred_val_upper, ' k-');
