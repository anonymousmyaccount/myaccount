clear all;
close all;
tic;
y = xlsread("beer.csv");
win = 12;
s1 = 0.0158*2;
kerfPara = struct('type','rbf','pars',s1);
C = 2^8;
[X_all,y_all] = build_dataset(y,win);
n_total = size(X_all,1);
splitIdx = floor(n_total*0.7);
X_train = X_all(1:splitIdx,:);
y_train = y_all(1:splitIdx,:);
X_test = X_all(splitIdx+1:end,:);
y_test = y_all(splitIdx+1:end,:);
[PredictY,f1,sparsity] = leastsquaresvr(X_train,y_train,X_test,kerfPara,C);
sigma = std(y_test-f1);
q = norminv([0.025 0.975],0,sigma);
Low = f1 + q(1);
Up = f1 + q(2);
PICP = mean((y_test>=Low) & (y_test<=Up));
MPIW = mean(Up-Low);
elapsed = toc;
fprintf('sparsity: %.4f\n',sparsity);
fprintf('sigma: %.4f\n',sigma);
fprintf('PICP: %.4f, MPIW: %.4f\n',PICP,MPIW);
fprintf('Time: %.2f sec\n',elapsed);
figure;
plot(y_test,'b-'); hold on;
plot(Low,'r-');
plot(Up,'k-'); hold off;
legend('True Values','Predicted Lower','Predicted Upper');
title('Prediction Intervals using leastsquaresvr');
function [X_all,y_all] = build_dataset(y,win)
n = length(y);
m = n - win;
X_all = zeros(m,win);
y_all = zeros(m,1);
for i = 1:m
    X_all(i,:) = y(i:i+win-1)';
    y_all(i) = y(i+win);
end
end