function [R t] = lc_syntheticCamera(distanceToCentre,fleche,sigmaTrans,up)

if nargin < 3 || isempty(sigmaTrans)
    sigmaTrans = 0;
end

if nargin < 4
    up = 0;
end

phiR = 2*pi*rand;
thetaR = fleche*pi/180;
%thetaR = 30*pi/180;
vec = [ sin(thetaR)*cos(phiR) sin(thetaR)*sin(phiR) cos(thetaR) ];

if up % 0 by default
    %vec = vec/norm(vec);
    %R(:,2) = cross(vec,[1;0;0]);
    %R(:,1) = cross(R(:,2,i),vec);
    %R(:,3) = vec;
else
    %vec = vec/norm(vec);%*rand()*2*pi;FAUX!!
end
%R = rotationMatrix( vec );

vec = vec';
vec = unit(-vec);
hyperplaneVec = null(vec*vec');
up = hyperplaneVec(:,1);
r2 = cross(vec,up);
r2 = unit(r2);
r1 = cross(r2,vec);
r1 = unit(r1);
R = [r1 r2 vec]';

% Rotation around optical center
psi = rand*2*pi;
%R = (R'*[cos(psi) -sin(psi) 0 ; sin(psi) cos(psi) 0 ; 0 0 1 ])';

% Compute translation part.

% On place la camera � une distance distanceToCentre de l'origine.

%t = [ distanceToCentre/10*randn ; distanceToCentre/10*rand ;  distanceToCentre ]
t = [ distanceToCentre/40*randn ; distanceToCentre/40*randn ;  distanceToCentre ]

%caux = -R'*t;
%caux = caux + sigmaTrans*(rand(3,1)-ones(3,1))/2;

%t = -R*caux;

display_ = 0;

if display_
    drawcamera([R t]);
    hold on;
    drawCoordinateSystem;
    axis equal;
    axis([-5 5 -5 5 0 20]);
end