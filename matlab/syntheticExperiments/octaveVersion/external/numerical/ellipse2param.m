function X = ellipse2param(ellipse)
% ellipse =
%   [     a, 1/2*b, 1/2*d; 
%     1/2*b,     c, 1/2*e;
%     1/2*d, 1/2*e,     f ];

a   = ellipse(1,1);
b   = 2*ellipse(1,2);
c   = ellipse(2,2);
d   = 2*ellipse(1,3);
e   = 2*ellipse(2,3);
f   = ellipse(3,3);
par = [ a; b; c; d; e; f];

thetarad    = 0.5*atan2(par(2),par(1) - par(3));
%thetarad = atan(par(2)/(sqrt((par(1) - par(3))^2)+par(2)));

cost        = cos(thetarad);
sint        = sin(thetarad);
sin_squared = sint.*sint;
cos_squared = cost.*cost;
cos_sin     = sint .* cost;

Ao      = par(6);
Au      = par(4) .* cost + par(5) .* sint;
Av      = - par(4) .* sint + par(5) .* cost;
Auu     = par(1) .* cos_squared + par(3) .* sin_squared + par(2) .* cos_sin;
Avv     = par(1) .* sin_squared + par(3) .* cos_squared - par(2) .* cos_sin;

if Auu == 0 | Avv == 0 
    
   X = zeros(1,5);
    
else
       
    tuCentre = - Au./(2.*Auu);
    tvCentre = - Av./(2.*Avv);
    wCentre  = Ao - Auu.*tuCentre.*tuCentre - Avv.*tvCentre.*tvCentre;

    uCentre = tuCentre .* cost - tvCentre .* sint;
    vCentre = tuCentre .* sint + tvCentre .* cost;

    Ru      = -wCentre./Auu;
    Rv      = -wCentre./Avv;

    Ru      = sqrt(abs(Ru)).*sign(Ru);
    Rv      = sqrt(abs(Rv)).*sign(Rv);

    X       = [ uCentre, vCentre, Ru, Rv, thetarad ];

end 
