clc; clear all

%% Implementation on Newton Method for HW 5 9.30
% by LO, Li-yu
% 13/May/2023

%% pre-settings
randn('state',1);
m=100;
n=40;

A = randn(m,n);

ITERATEMAX = 1000;
alpha = 0.25; beta = 0.5;
ita = 1e-4;

% objective
objval = inf; 

% variable
x = zeros(n,1);

% gradient
grad = []; %  d(f(x))/dx = A'*(1./(1-A*x)) - 1./(1+x) + 1./(1-x);

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
    
    objval = -sum(log(ones(m,1) - A*x)) - sum(log(ones(n,1)-x.*x));
    
    all_objval(k,1) = objval;
    
    % get gradient
    
    grad = A'*(1./(1-A*x)) - 1./(1+x) + 1./(1-x);
    hess =  A'*diag((1./(1-A*x)).^2)*A + diag(1./(1+x).^2 + 1./(1-x).^2);
    
    deltax = -hess^(-1) * grad;
    
    lamdasqr = grad' * deltax;
    
        
    % check termination criterion
    if abs(lamdasqr) <= ita
        break;
    end
    
    % backtracking line search for step size
    t = t0;
  
    while true 
        xkk = x + t * deltax;
        Axkk = A * xkk;
        
        if (max(Axkk) >= 1) || (max(abs(xkk)) >= 1) % check x \in domf
            t = beta * t;
        else
            break;
        end

    end
        
    while true 
        xkk = x + t * deltax;
        
        objvalkk = -sum(log(1-A*(xkk))) - sum(log(1-(xkk).^2));
        fray = objval + alpha * t * grad' * deltax;
        
        if objvalkk <= fray
            break;
        else
            t = beta * t;
        end
    end
    x = x + t * deltax;
    
    all_ts(k,1) = t;
end



objval_final = objval;


disp("END!");


% %% figures
% 
figure(1)
plot(all_k, all_objval, '-');
xlabel('k'); ylabel('f(x)');
ax = gca;


figure(2) %step length
plot(all_k(1:length(all_ts)), all_ts,':', all_k(1:length(all_ts)), all_ts, 'o');
xlabel('k'); ylabel('t');



figure(3)
semilogy(all_k, all_objval - objval_final, '-');
xlabel('k'); ylabel('f(x) - p*');


