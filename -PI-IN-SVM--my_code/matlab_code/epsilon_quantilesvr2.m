function [PredictY,f1,nsv,sparsity] = epsilon_quantilesvr2(X,Y,test,kerfPara,C,tau,eps1) 
    % tolerance for Support Vector Detection
     epsilon = svtol(C);
     n = size(X,1);
    % Construct the Kernel matrix
    fprintf('Constructing ...\n');
    H = zeros(n,n);  
    H= kernelfun(X,kerfPara);
    
    % Set up the parameters for the Optimisation problem
        Hb = [H -H; -H H];
        c = [((1-tau)*eps1*ones(n,1) - Y); (tau*eps1*ones(n,1) + Y)];  
        vlb = zeros(2*n,1);    % Set the bounds: alphas >= 0
        vub = [tau*C*ones(n,1);((1-tau)*C)*ones(n,1)];   %                 alphas <= C
        x0 = zeros(2*n,1);     % The starting point is [0 0 0   0]
        neqcstr = nobias(kerfPara.type); % Set the number of equality constraints (1 or 0)  
        if neqcstr
          A = [ones(1,n) -ones(1,n)];, b = 0;     % Set the constraint Ax = b
        else
          A = [];, b = []; 
        end
    
    % Add small amount of zero order regularisation to 
    % avoid problems when Hessian is badly conditioned. 
    % Rank is always less than or equal to n.
    % Note that adding to much reg will peturb solution
    Hb = Hb+1e-10*eye(size(Hb));
    % Solve the Optimisation Problem
   % fprintf('Optimising ...\n');
    st = cputime;
  %  options= optimset('Algorithm','interior-point-convex','MaxIter',1000,'TolFun',1e-4);
    [alpha, lambda ,how] = quadprog(Hb, c, [], [], A,b,vlb, vub, x0);
   % fprintf('Execution time : %4.1f seconds\n',cputime - st);
    %fprintf('Status : %s\n',how);
    alpha1= alpha(1:n);
    beta1= alpha(n+1:2*n);
     beta =  alpha1-beta1;
   % fprintf('|w0|^2    : %f\n',beta'*H*beta);  
   % fprintf('Sum beta : %f\n',sum(beta));
    % In epsilon_quantilesvr2 function
    %fprintf('Number of non-zero beta entries: %d\n', nnz(beta));
    %fprintf('Total number of beta entries: %d\n', length(beta));
    %fprintf('Sparsity percentage: %f%%\n', 100 * nnz(beta) / length(beta));
        
    %% Compute the number of Support Vectors
%       error1= find((alpha1>((C)*tau)-epsilon) & (alpha1<((C)*tau)+epsilon));
%      error2= find((beta1>((C)*(1-tau))-epsilon) & (beta1<((C)*(1-tau))+epsilon));
%      sv1i = find((alpha1 > epsilon) & (alpha1 < (((C)*tau)-epsilon)));
%      sv2i=find((beta1 > epsilon) & (beta1 < (((C)*(1-tau))-epsilon)));
%      out.sv1= length(sv1i)+length(error1);
%      out.sv2= length(sv2i)+length(error2);
%      out.err1= length(error1);
%      out.err2=length(error2);
%      out.eps1=eps1;
%     
    
 %%   
    
   % Implicit bias, b0
    bias = 0;
    sparsity = 1- ((nnz(beta))/length(beta));
    % Explicit bias, b0 
    
    if nobias(kerfPara.type) ~= 0
      
          % find bias from average of support vectors with interpolation error e
          % SVs with interpolation error e have alphas: 0 < alpha < C
          
          if(tau>0.5)
          svii = find( abs(beta) > epsilon & abs(beta) < (tau*C - epsilon));
          else
              svii = find( abs(beta) > epsilon & abs(beta) < ((1-tau)*C - epsilon));
          end
          
          
          if length(svii) > 0
            bias = (1/length(svii))*sum(Y(svii) - e*sign(beta(svii)) - H(svii,svi)*beta(svi));
          else v
            fprintf('No support vectors with interpolation error e - cannot compute bias.\n');
            bias = (max(Y)+min(Y))/2;
          end
        
    end
      Htest=kernelfun(test,kerfPara,X);
      f1 = H*beta + bias;
      
      PredictY = Htest*beta + bias;
      nsv=length(find(abs(beta)>epsilon));
end