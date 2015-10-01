function [ IResult, homography ] = lc_generateRandomHomographieAux( I, distance, angle, focale, sigmaNoise, lenghtMotionBlur, occlusion )

%distance = rand*40 + 10;
%angle = 2*(rand-0.5)*angle;

% Occultation
%I = generateOcclusion(occlusion);

[I , rRes] = generateOcclusion(I, occlusion);

%imagesc(I);colormap gray;
%axis equal;


% 1280*720
[R t] = lc_syntheticCamera(distance,angle,0);

K = [ focale 0 640
      0 focale 360
      0 0  1 ];

% Projection matrix
Projection = K*[R t];

% From image marker pixel coordinate to unit
N = vgg_conditioner_from_image(832,832,320);
%N = eye(3);%

% Compute homography
homographyAux = K*[ R(:,1:2) t ];
homography = K*[ R(:,1:2) t ]*N;
ptsRef = [ 0 0 ; 1 0 ; 1 1 ; 0 1 ];

imPtsRef = normalize(homography*augment(ptsRef'));

T = maketform('projective',homography');

[IResult,xdat,ydat] = imtransform(I,T,'bilinear','fill',255);


% Flou de profondeur de champs
%G = fspecial('gaussian',[5 5],0.6);
%IResult = imfilter(IResult,G,'replicate');

% Flou de bougé
%M = fspecial('motion', ??, ??);
%IResult = imfilter(IResult,M,'replicate');

% Add noise
IResult = double(IResult) + sigmaNoise*randn(size(IResult));

IResult = IResult/1.5+40;

% Flou de mise au point
G = fspecial('gaussian',[5 5],0.5);
IResult = imfilter(IResult,G,'replicate');


% Flou de bougé
if lenghtMotionBlur > 0
    M = fspecial('motion',lenghtMotionBlur, 0 );
    IResult = imfilter(IResult,M,'replicate');
end

IResult = mat2gray(IResult,[0 255]);
%pause

%IResult = IResult*255;
%IResult(40:60,40:60)
%pause

%figure;
%imagesc(IResult);colormap gray
%pause

if 0
    Iaux = max(max(IResult))*ones(720,1280);
    Iaux(ydat(1):ydat(2),xdat(1):xdat(2)) = IResult;
    IResult = Iaux;
    
   imagesc(IResult); colormap gray;hold on;
   pause
   %ind = find(abs(round(IResult) - 160) <= 4);
   
   GX = fspecial('sobel');
   GY = GX';
   Idx = conv2(IResult,GX,'same');
   Idy = conv2(IResult,GY,'same');
   
   Ing = sqrt(Idx.*Idx + Idy.*Idy);
   ind = find(abs(round(Ing) - 50) <= 4);
   
   for k=320:-5:30
       % Generate points to display in debug mode
       Q = param2ellipse([0 0 k k 0]);
       E = homography'\ Q / homography;
       par1 = ellipse2param(E);
       pts1 = ellipsepoints(par1,100);
       plot(pts1(1,:),pts1(2,:),'g');
   end
   
   %length(ind)
   %pause
[x,y] = ind2sub(size(IResult),ind);
plot(y,x,'r.');
   pause
end

debug_ = 0;

if 0%debug_
    figure
    imagesc(double(IResult));%axis([0 1280 0 720]);
    colormap gray;
    pause
end

rhoOld = 0;

if debug_
    
    % Generate points to display in debug mode
    
    figure; axis equal; hold on;
    for i=1:-0.1:0
        
        co = [ i 0.5 0.5 ];
        
    Q = param2ellipse([0 0 i i 0]);
    E = homographyAux'\ Q / homographyAux;
    par1 = ellipse2param(E);
    pts1 = ellipsepoints(par1,100);
    %Q = param2ellipse([0 0 0.6 0.6 0]);
    %E = homographyAux'\ Q / homographyAux;
    %par2 = ellipse2param(E);
    %pts2 = ellipsepoints(par2,100);
    %Q = param2ellipse([0 0 0.2 0.2 0]);
    %E = homographyAux'\ Q / homographyAux;
    %par3 = ellipse2param(E);
    %pts3 = ellipsepoints(par3,100);
    
    %figure;hold on;
    plot(pts1(1,:),pts1(2,:),'Color',co,'LineWidth',2);
    %plot(pts2(1,:),pts2(2,:),'g+');
    %plot(pts3(1,:),pts3(2,:),'b+');
    %axis equal;
    %pause
    
    plot( par1(1),par1(2), 'Color', co, 'Marker','*','MarkerSize',10 );
    
    %par1(1)
    %par1(2)
    
    %par2(1)
    %par2(2)
    
    % Display small semi-axis direction
    xb = par1(1)+cos(par1(5))*par1(3);
    yb = par1(2)+sin(par1(5))*par1(3);
    plot(xb,yb,'b.','MarkerSize',10);
    xb = par1(1)-cos(par1(5))*par1(3);
    yb = par1(2)-sin(par1(5))*par1(3);
    plot(xb,yb,'b.','MarkerSize',10);
    
    line([par1(1) par1(1)+cos(par1(5))*600], [ par1(2) par1(2)+sin(par1(5))*600 ]); 
    
    sin(par1(5)-rhoOld)
    %mod(par1(5)-rhoOld,2*pi)
    rhoOld = par1(5);
    
    %xb = par1(1)-cos(par1(5))*par1(3);
    %yb = par1(2)-sin(par1(5))*par1(3);
    %plot(xb,yb,'b.','MarkerSize',10);
    
    %xb = par2(1)+cos(par2(5))*par2(3);
    %yb = par2(2)+sin(par2(5))*par2(3);
    %plot(xb,yb,'b.','MarkerSize',30);
    %xb = par3(1)+cos(par3(5))*par3(3);
    %yb = par3(2)+sin(par3(5))*par3(3);
    %plot(xb,yb,'b.','MarkerSize',30);
    end
    
    imOrig = normalize( Projection*[0;0;0;1] );
    imN3 = normalize( Projection*[0;0;1;1] );
    
    h = line([imOrig(1) imN3(1)],[imOrig(2) imN3(2)]);
    
    plot(imOrig(1),imOrig(2),'ro');
    
    %plot( par1(1),par1(2), 'r*' );
    %plot( par2(1),par2(2), 'g*' );
    %plot( par3(1),par3(2), 'b*' );
    
end

