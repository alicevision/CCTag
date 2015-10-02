function C = param2ellipse(param)
if length(param)==6, 
   error('For ellipse Coefficients to matrix: use coeffs2ellipse');
end
% param  = [ xc;yc;ax;bx;rho ]
xc    = param(1);
yc    = param(2);
ax    = param(3);
bx    = param(4);
rho   = param(5);
T     = [ cos(rho),-sin(rho),xc; sin(rho),cos(rho),yc; 0,0,1 ];
C     = transpose(inv(T))*diag([1/ax^2;1/bx^2;-1])*inv(T);
% C     = C/norm(C,'fro');
