clc; clear all

%% Implementation on Gradient Descent Method for HW 5 9.30
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
objval = inf; %  f(x) = sum(-log(ones(n,0) - A*x) - sum(-log(ones(n,0)-x.*x))

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

%% backtracking line search analysis
ks = [];
objval_finals = [];
alphas = [];
alpha = 0.01;
counter = 1;

for alpha = 0.01 :0.01 : 0.49
x = x * 0;

%% Gradient Descent Method
for k = 1:ITERATEMAX
    % record
    all_k(k,1) = k;
    
    objval = -sum(log(ones(m,1) - A*x)) - sum(log(ones(n,1)-x.*x));
    
    all_objval(k,1) = objval;
    
    % get gradient
    grad = A'*(1./(1-A*x)) - 1./(1+x) + 1./(1-x);
    deltax = -grad;
    
        
    % check termination criterion
    if norm(grad) <= ita
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

objval_finals(counter,1) = objval;
ks (counter,1) = k;
alphas(counter,1) = beta;
counter = counter + 1;

disp(counter);




end

disp("END");

figure(10)
plot([1:length(ks)], ks)
xlabel('x 0.01 alpha'); ylabel('k');

% xlim([-0.1 1.1])


figure(20)
plot([1:length(ks)], objval_finals)
xlabel('x 0.01 alpha'); ylabel('f(x)');
ylim([-49.04 -49.028])



% %% figures
% 
% figure(1)
% plot(all_k, all_objval, '-');
% xlabel('k'); ylabel('f(x)');
% 
% figure(2) %step length
% plot(all_k(1:length(all_ts)), all_ts,':', all_k(1:length(all_ts)), all_ts, 'o');
% xlabel('k'); ylabel('t');
% 
% 
% figure(3)
% semilogy(all_k, all_objval - objval_final, '-');
% xlabel('k'); ylabel('f(x) - p*');
% 
% % figure(3)
% % plot([0:(length(vals)-2)], vals(1:length(vals)-1))

