function [PredictY,f1,sparsity] = leastsquaresvr(X,Y,test,kerfPara,C)
      n = size(X,1);
      e= ones(n,1);
      fprintf('Constructing ...\n');
      H= kernelfun(X,kerfPara)+ (1/C).*eye(n,n);
      R = [0 ,e' ; e  , H];%+(1/C).*eye(n,n)];
      balpha =  inv(R)*[0;Y];
      b= balpha(1);
     alpha= balpha(2:end);
     f1 = H*alpha+b;
     PredictY = kernelfun(test,kerfPara,X)*alpha + b;  
     sparsity = 1- ((nnz(alpha))/length(alpha))
  end
