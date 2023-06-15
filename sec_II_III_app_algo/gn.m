clc; clear all

%% Implementation on Gauss-Newton Method for HW 5 9.32
% by LO, Li-yu
% 14/May/2023

%% pre-settings
n = 40;
m = 100;
As = cell(1,m);
bs = cell(1,m);

for i = 1:m
    while true
        A = rand(n,n);

        if rank(A) == n
            break;
        end
    end
    A = A'* A;
    As{i} = A;
    
    b = rand(n,1);
    b = b / (b'* A^(-1) * b / 1.95)^(1/2);
    bs{i} = b;
    
end



ITERATEMAX = 1000;
alpha = 0.01; beta = 0.5;
ita = 1e-4;

% objective
objval = inf; 
% variable
x = zeros(n,1);

% gradient
grad = []; 

% stepl
t0 = 1;

% misc
all_k = [];
all_objval = [];
all_ts = [];
objval_final = objval;

%% Gauss-Newton Method
for k = 1:ITERATEMAX
    % record
    all_k(k,1) = k;
    
    fx = 0;
    grad = zeros(n,1);
    gradsquare = zeros(n,n);
    
    for i = 1:m
        Ai = As{i};
        bi = bs{i};
        
        % objective
        fxi = 0.5 * x' * Ai * x + bi' * x + 1;
        fx = fx + fxi^2;
        
        gradi = fxi * (Ai * x + bi);
        grad = grad + gradi;
        
        gradsquarei =(Ai * x + bi) * (Ai * x + bi)';
        gradsquare = gradsquare + gradsquarei;
        temp = 0;
    end
    
    fx = 0.5 * fx;
    
    objval = fx;
    all_objval(k,1) = fx;
    
    dgn = -1 * gradsquare^(-1) * grad;
    
    
        
    % check termination criterion
    if norm(dgn) <= ita
        break;
    end
    
    % backtracking line search for step size
    t = t0;
 
        
    while true 
        xkk = x + t * dgn;
        
        fxkk = 0;
        for i = 1:m
            Ai = As{i};
            bi = bs{i};
            fxikk = 0.5 * xkk' * Ai * xkk + bi' * xkk + 1;
            fxkk = fxkk + fxikk^2;
        end
        fxkk = 0.5 * fxkk;
        
        fray = objval + alpha * t * grad' * dgn;
        
        if fxkk <= fray
            break;
        else
            t = beta * t;
        end
    end
    x = x + t * dgn;    
    all_ts(k,1) = t;           
end


%%

objval_final = objval;


disp("END!");

% %% figures
% 
figure(1)
semilogy(all_k, all_objval - objval_final, '-');
xlabel('k'); ylabel('f(x) - p*');
