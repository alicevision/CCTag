function xp = lc_loadVaryingParameters( iXp )

if (iXp == 1)
    xp.nTest = 10;
    xp.distance = [ 10 20 30 40 ];%10 15 20 25 30 35 40 ];
    xp.fixedDistance = 1; % 1: the camera will be placed at the specified distance
                       % otherwise between the specified distance and 0.
    xp.angle = [ 0 25 50 75 ];
    xp.fixedAngle = 1; % 1: the camera will be placed at the specified angle
                    % otherwise between the specified angle and 0.
    xp.focale = [ 1000 ];
    xp.focalRange = 0; % 1: the camera will have the specified focale
                       % otherwise a focal whose the value is located between 
                       % the specified focal + [- focalRange ; focalRange ].
    xp.sigmaNoise = [ 5 ]; % 0 5 10 15
    xp.lengthMotionBlur = [ 0 ];
    xp.occlusion = [ 0 ];
elseif (iXp == 2)
    xp.nTest = 10;
    xp.distance = [ 10 20 30 40 ];%[ 10 20 30 40 50 ];
    xp.fixedDistance = 1;
    xp.angle = [ 20 ];
    xp.fixedAngle = 0;
    xp.focale = [ 1000 ];
    xp.focalRange = 0;
    xp.sigmaNoise = [ 5 ]; % 0 5 10 15
    xp.lengthMotionBlur = [ 0 6 12 18 ]; %[ 0 2 4 6 8 10  ];
    xp.occlusion = [ 0 ];
elseif (iXp == 3)
    xp.nTest = 10;
    xp.distance = [ 10 20 30 40 ];%50
    xp.fixedDistance = 1;
    xp.angle = [ 60 ];
    xp.fixedAngle = 0;
    xp.focale = [ 1000 ];
    xp.focalRange = 0;
    xp.sigmaNoise = [ 5 ]; % 0 5 10 15
    xp.lengthMotionBlur = [ 0 ];
    xp.occlusion = [ 0 0.25 0.5 0.7 ];
end