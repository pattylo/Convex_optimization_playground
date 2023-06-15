clc; clear all
tic;

%% Implementation on Newton Method for HW 5 10.15a
% by LO, Li-yu
% 14/May/2023

%% pre-settings
randn('state',1);
n=100;
p=30;

for i = 1:p
    while true
        A = rand(p,n);

        if rank(A) == p
            break;
        end
    end    
end

ITERATEMAX = 1000;
alpha = 0.25; beta = 0.5;
ita = 1e-4;

% objective
objval = inf; 

% variable
x = 0 + (1-0).*rand(n,1);

% gradient 
grad = []; %  d(f(x))/dx = A'*(1./(1-A*x)) - 1./(1+x) + 1./(1-x);

% constraint
b = A*x;

% stepl
t0 = 1;


% misc
all_k = [];
all_objval = [];
all_ts = [];
objval_final = objval;

%% Newton Method
for k = 1:ITERATEMAX
    % record
    all_k(k,1) = k;
    
    objval = x' * log(x);
    
    all_objval(k,1) = objval;
    
    % get gradient and hessian
    grad = log(x) + 1;
    hess = diag(1./x);
    
    % solve the linear equation to get newton step
    KKT_MAT = [hess, A'; A, zeros(p,p)];
    bb = [-grad;zeros(p,1)];
    
    xx = KKT_MAT^(-1) * bb;
    
    dxnt = xx(1:n);
    lamdasqr = grad' * dxnt;
    
        
    % check termination criterion
    if abs(lamdasqr) <= ita
        break;
    end
    
    % backtracking line search for step size
    t = t0;
        
    while true 
        xkk = x + t * dxnt;        
        objvalkk = xkk' * log(xkk);
        fray = objval + alpha * t * grad' * dxnt;
        
        if objvalkk <= fray
            break;
        else
            t = beta * t;
        end
    end
    x = x + t * dxnt;
    
    all_ts(k,1) = t;
end



objval_final = objval;
timelapsed = toc;

disp("END!");

figure(1)
semilogy(all_k, all_objval - objval_final, '-');
xlabel('k'); ylabel('f(x) - p*');


