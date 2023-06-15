clc; clear all
tic;

%% Implementation on Barrier Method for HW 5 10.15a
% by LO, Li-yu
% 14/May/2023

%% pre-settings

n = 2;
m = 5;
ITERATEMAX = 1000;

alpha = 0.25;
beta = 0.5;

ita = 1e-4; 
mu = 2;
epsilon = 1e-4; 

A = [0 -1;
     2 -4;
     2 1 ;
    -4 4 ;
    -4 0 ;];
b = 1;


Ap = max(A,0); 
An = max(-A,0);

r = max(Ap*ones(n,1) + An*ones(n,1));
u = (.5/r)*ones(n,1); l = -(.5/r)*ones(n,1);
t = 1;

%% Barrier Method
for iter = 1:ITERATEMAX
    
    y = b + An * l - Ap * u;
    val = -t*sum(log(u-l)) - sum(log(y));
    
    grad = t*[1./(u-l); -1./(u-l)] + [-An'; Ap']*(1./y);
    hess = t*[diag(1./(u-l).^2), -diag(1./(u-l).^2);
        -diag(1./(u-l).^2), diag(1./(u-l).^2)] + ...
        [-An'; Ap']*diag(1./y.^2)*[-An Ap];
    
    step = -hess\grad; 
    lamdasqr = grad'*step;
    
    if (abs(lamdasqr) < ita)
        gap = m/t;
        if (gap < epsilon) 
            break; 
        end
        t = mu*t;
    else
        dl = step(1:n); du = step(n+[1:n]); 
        dy = An*dl-Ap*du;
        tls = 1;
        while (min([u-l+tls*(du-dl); y+tls*dy]) <= 0)
            tls = beta*tls;
        end
        
        while (-t*sum(log(u-l+tls*(du-dl))) - sum(log(y+tls*dy)) >= val + tls*alpha*lamdasqr)
            tls = beta*tls;
        end
        
        l = l+tls*dl; u = u+tls*du;
    end
    
end

disp("%%%%%%");
disp(u);
disp("%%%%%%");
disp(l);
disp("END");

timeElapsed = toc;

A = [0 -1;
     2 -4;
     2 1 ;
    -4 4 ;
    -4 0 ;];
b = 1;

%% plot

figure(1)
for i = 1:m 
    ai = A(i,1);
    bi = A(i,2);
    ci = b;
    
    if bi == 0
        yi = -0.5:0.01:0.5;
        xi = (-bi * yi + ci)/ai;
        plot(xi,yi);
        gan = 0;
        
    else
        xi = -0.5:0.01:0.5;
        yi = (-ai * xi + ci)/bi;
        plot(xi,yi);
    end
    
    hold on
    
end
xlim([-0.5 0.5])
ylim([-0.5 0.5])
hold on


xi = -0.5:0.01:0.5;
yi = u(2,1) * ones(1, length(xi));
plot(xi,yi,'--');

xi = -0.5:0.01:0.5;
yi = l(2,1) * ones(1, length(xi));
plot(xi,yi,'--');

yi = -0.5:0.01:0.5;
xi = u(1,1) * ones(1, length(xi));
plot(xi,yi,'--');

yi = -0.5:0.01:0.5;
xi = l(1,1) * ones(1, length(xi));
plot(xi,yi,'--');




