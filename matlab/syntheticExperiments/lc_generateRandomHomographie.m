function [ vImagePattern, homography, metricHomography] = lc_generateRandomHomographie( vPatterns, distance, fixedDistance, ...
    angle, fixedAngle, focale, focalRange, sigmaNoise, lenghtMotionBlur, occlusion, display_)

if ( nargin < 11 )
    display_ = 0;
end
%display_ = 1;

radiusPixel = 400;

[height, width] = size(vPatterns(1).I);

%[I , rRes] = generateOcclusion(I, occlusion, width, height, radiusPixel);

% Generate occlusions
for iPattern=1:length(vPatterns)
    vPatterns(iPattern).I = lc_generateOcclusion(vPatterns(iPattern), occlusion, radiusPixel);
end

if display_
    figure; imagesc(I);colormap gray;
    axis equal;
    pause
end

% 1280*720
distMin = 10;
distMax = 50;
if ~fixedDistance
    distance = distMin + (distMax-distMin)*rand;
end

angleMin = 0;
angleMax = 70;
if ~fixedAngle
    angle = angleMin + (angleMax-angleMin)*rand;
end

angle

[R t] = lc_syntheticCamera(distance,angle);

focale = focale - focalRange + 2*focalRange*rand;

K = [ focale 0      640
      0      focale 360
      0      0      1   ];

% Projection matrix
Projection = K*[R t];

% From image marker pixel coordinate to unit
N = vgg_conditioner_from_image(width,height,radiusPixel);
%N = eye(3);%

% Compute homography
metricHomography = K*[ R(:,1:2) t ];
homography = K*[ R(:,1:2) t ]*N;
ptsRef = [ 0 0 ; 1 0 ; 1 1 ; 0 1 ];

imPtsRef = normalize(homography*augment(ptsRef'));

T = maketform('projective',homography');

% Noise component
noise = sigmaNoise*randn(720,1280);

% Motion blur kernel
if lenghtMotionBlur > 0
    M = fspecial('motion',lenghtMotionBlur, 180*rand );
end

% Out-of-focus kernel
G = fspecial('gaussian',[5 5],0.5);

for iPattern=1:length(vPatterns)
    [IResult,xdat,ydat] = imtransform(vPatterns(iPattern).I,T,'bilinear','fill',255);
    
    aux = 255*ones(720,1280);
    aux(ydat(1):ydat(2),xdat(1):xdat(2)) = IResult;
    IResult = aux;
    
    if display_
        figure;
        imagesc(IResult);
        axis equal; axis off; colormap gray; hold on;
        imCenter = normalize(metricHomography*[0;0;1]);
        plot(imCenter(1),imCenter(2),'b*');
        pause
    end
    
    % Flou de profondeur de champs
    %G = fspecial('gaussian',[5 5],0.6);
    %IResult = imfilter(IResult,G,'replicate');
    
    % Flou de boug�
    %M = fspecial('motion', ??, ??);
    %IResult = imfilter(IResult,M,'replicate');
    
    % Add noise
    IResult = double(IResult) + noise;
    
    IResult = IResult/1.5+40;
    
    % Flou de mise au point
    IResult = imfilter(IResult,G,'replicate');
    
    
    % Flou de boug�
    if lenghtMotionBlur > 0
        IResult = imfilter(IResult,M,'replicate');
    end
    
    IResult = mat2gray(IResult,[0 255]);
    
    vImagePattern(iPattern) = vPatterns(iPattern);
    vImagePattern(iPattern).I = IResult;
    
    
    %subplot(1,2,iPattern);
    %imagesc(IResult);colormap gray; axis equal;axis off;
    %pause
end
%pause
%close all;


