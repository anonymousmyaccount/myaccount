function[trainf1,f1,sparsity]=quantileLPONENORMTSVR(train ,ytrain,test,s,c3,c1,tau1)
n1 = length(train);
n2 = length(test);
e = ones(n1,1);
kerfPara.type = 'rbf';
kerfPara.pars=s;
A = kernelfun(train,kerfPara);
H=[A,e]; 
%% Optimization Problem -1
%     r1           p1              xi_1      xi2
f = [c3*ones(n1+1,1); c3*ones(n1+1,1); c1*tau1*e ;    c1*(1-tau1)*e];
A1 = [-H  ,            H,        -eye(n1,n1),  zeros(n1,n1);
      H  ,            -H,          zeros(n1,n1) ,-eye(n1,n1)];                                                     
B1 = [-ytrain;ytrain];
lb1 = [zeros(n1+1,1);zeros(n1+1,1);zeros(n1,1);zeros(n1,1)];
ub1 = [inf(n1+1,1);inf(n1+1,1);inf(n1,1);inf(n1,1)];
options = optimoptions('linprog', 'Display', 'off'); % Suppress output
[x, ~, exitflag] = linprog(f, A1, B1, [], [], lb1, ub1, options);
%[x1,~,exitflag]= linprog(f,A1,B1,[],[],lb1,ub1);
Htest = [kernelfun(test,kerfPara,train),ones(size(test,1),1)];
exitflag
u1 =x1(1:n1+1,:)-x1(n1+2:2*(n1+1),:);
trainf1 = H*u1;
f1 = Htest*u1;
sparsity = sum(abs(u1) < 0.0001 )/ length(u1)*100;
end