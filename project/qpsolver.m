%% Implementation on Newton Method 
%% with Equality Constraint and Infeasiblity Start 
%% for mini-project
% by LO, Li-yu
% 20/May/2023

function polycoeff = qpsolver(Q_0, Aeq, beq)
 
[p,n] = size(Aeq);
 
iteration_max = 100;
alpha = 0.01;
beta = 0.5;
 
eps = 1e-6;
x=10000 * ones(n,1);  
nu=zeros(p,1);

objval = [];

all_step = [];
 
for i=1:iteration_max
    
   r = [Q_0 * x + Aeq'*nu;  Aeq*x-beq];  
   sol = -inv([Q_0 Aeq';  Aeq zeros(p,p)]) * r;
   
   objval = [objval x'*Q_0*x];
 
   Dx = sol(1:n);  
   Dnu = sol(n+1:end);
   
   if (norm(r) < eps) 
       disp("Aeqx=beq!!!!!");
       break; 
   end
 
   t=1;
 
   while norm([Q_0 * (x+t * Dx) + Aeq'*(nu+t*Dnu);  Aeq*(x+t*Dx)-beq])  > (1-alpha*t)*norm(r)
        t=beta*t; 
   end
 
   x = x + t * Dx; 
   nu = nu + t * Dnu;
   all_step = [all_step t];
end
 
final_val = x'*Q_0*x;
 
polycoeff = x;
 
 
