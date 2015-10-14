% fileName is the name of the file to parse, e.g. path/data.txt
% imagePath is the path of the folder containing the image(s) associated
% to the parsed markers
% display_ true to display parsed data

function allMarkers = parseCCTags(fileName, imagePath, display_)

if nargin < 3
    display_ = 0;
end

fid = fopen(fileName);

new = fscanf(fid, '%d', 1);

i = 1;

imageNameFile = [ imagePath '/viewLevel0.png' ];
imageVoteNameFile = [ imagePath '/voteLevel0.png' ];
imageCannyNameFile = [ imagePath '/cannyLevel0.png' ];

while ( exist(imageCannyNameFile, 'file') > 0 )
    I{i} = imageNameFile;%imread(imageNameFile);
    %IVoteAux = imread(imageVoteNameFile);
    
    ING{i} = imageNameFile;%double(rgb2gray(I{i}));
    IVote{i} = imageVoteNameFile;%rgb2gray(IVoteAux);
    %[ sobelFilterX, sobelFilterY ] = lc_cctag_sobelFilter;
    
    IDX{i} = []; %conv2(ING{i},sobelFilterX, 'same');
    IDY{i} = []; %conv2(ING{i},sobelFilterY, 'same');
    
    ICanny{i} = imageCannyNameFile; %rgb2gray(imread(imageCannyNameFile));
    
    i = i+1;
    imageNameFile = [ imagePath '/viewLevel' int2str(i-1) '.png' ];
    imageVoteNameFile = [ imagePath '/voteLevel' int2str(i-1) '.png' ];
    imageCannyNameFile = [ imagePath '/cannyLevel' int2str(i-1) '.png' ];
end

% The maximum number of levels in the pyramid is supposed to be less than
% 10
for i=1:10
    pyramid{i} = [];
end

while ~isempty(new)
    
    % Skip string '22 serialization::archive 10 0 0'
    fscanf(fid, [' serialization::archive 10 0 0 ']);
    
    % Read nCircles
    marker.nCircles = fscanf(fid, '%d',1);
    
    % Read Id
    marker.id = fscanf(fid, '%d',1)+1;
    
    % Read pyramidLevel
    marker.pyramidLevel = fscanf(fid, '%d',1)+1;
    
    % Read scale
    marker.scale = fscanf(fid, '%f',1);
    
    % Read status
    marker.status = fscanf(fid, '%d',1);
 
    % Read outer ellipse
    marker.outerEllipse = readEllipse(fid);
    
    % Read rescaled outer ellipse
    marker.rescaledOuterEllipse = readEllipse(fid);
    
    if display_
        imshow(I{marker.pyramidLevel}); hold on;
        
        %imshow(IVote{marker.pyramidLevel}); hold on;
    end
    
    % Read pair idSet <id, probaIda>
    marker.idSet = readIdSet(fid);
    
    marker.type = 1;
    
    if ~isempty(marker.idSet)
        marker.qId = marker.idSet(1,2);
    end
    
    % Read radius ratio
    marker.radiusRatio = readRadiusRatio(fid);
    
    % Read quality
    marker.quality = fscanf(fid, '%f',1);
    
    % Read points
    marker.points = readPoints(fid);
    
    % Read ellipses
    marker.ellipses = readEllipses(fid);
    
    % Read homography
    marker.homography = normalize_matr_det_1(fscanf(fid, '%f', [3 3])); % TODO translate homography
    
    if display_
        %A = diag([ 1 1 -1]);
        %Q = inv(marker.homography)'*A*inv(marker.homography);
        %displayEllipse(Q,'r');
        %pause
    end
    
    % Read imgCenter
    marker.imgCenter = [fscanf(fid, '%f', [2 1]) ; 1] + [ 1 ; 1 ; 0 ];
    
    % Read flowComponents
    marker.flowComponents = readFlowComponents(fid, marker.nCircles);
    
    displayMarkers_ = 1;
    displayFlowComponents_ = 0;
    
    if display_
        ellipseColor = 'rgbmcyrgb';
        
        if displayFlowComponents_
            
            for k=1:length(marker.flowComponents)
                flowComponents = marker.flowComponents;
                flowComponent = flowComponents(k);
                
                outerEllipsePoints = flowComponent.outerEllipsePoints;
                quiver(outerEllipsePoints(:,1),outerEllipsePoints(:,2),outerEllipsePoints(:,3),outerEllipsePoints(:,4), 1, 'color', 'b');
                
                convexEdgeSegment = flowComponent.convexEdgeSegment;
                quiver(convexEdgeSegment(:,1),convexEdgeSegment(:,2),convexEdgeSegment(:,3),convexEdgeSegment(:,4), 1, 'color', 'y');
                
                displayEllipse(flowComponent.outerEllipse, 'r');
                
                fieldLines = flowComponent.fieldLines;
                for i=1:size(fieldLines,1)
                    %plot(fieldLines(i,:,1),fieldLines(i,:,2));
                    quiver(fieldLines(i,:,1),fieldLines(i,:,2),fieldLines(i,:,3),fieldLines(i,:,4), 0.2, 'color', 'm');
                end
                
                filteredFieldLines = flowComponent.filteredFieldLines;
                for i=1:size(filteredFieldLines,1)
                    %plot(fieldLines(i,:,1),fieldLines(i,:,2));
                    quiver(filteredFieldLines(i,:,1),filteredFieldLines(i,:,2),filteredFieldLines(i,:,3),filteredFieldLines(i,:,4), 0.2, 'color', 'g');
                end
                plot(flowComponent.seed(1), flowComponent.seed(2), 'go');
                
            end
        end
        
        % Display detected points and ellipses
        if displayMarkers_
            
            displayEllipse(marker.outerEllipse, 'b');
            
            for k=1:length(marker.points)
                pts = marker.points{k};
                if size(pts,1) ~= 0
                    %plot(pts(:,1),pts(:,2),'go');
                end
                if ~isempty(marker.ellipses)
                    ellipse = marker.ellipses{k};
                    displayEllipse(ellipse, 'g');
                end
            end
            plot(marker.imgCenter(1),marker.imgCenter(2),'rx');
        end
        
        pause
        hold off;
    end
    
    pyramid{marker.pyramidLevel} = [ pyramid{marker.pyramidLevel} marker ];
    
    % are there new markers ?
    new = fscanf(fid, '%d',1);
end

fclose(fid);

save( [ imagePath '/data.mat'], 'fileName', 'imagePath', 'I', 'ING', 'IVote', 'ICanny', 'IDX', 'IDY', 'pyramid');

if nargout > 0
    allMarkers = [];
    for i=1:10
        allMarkers = [ allMarkers pyramid{i} ];
    end
end

end

function ellipses = readEllipses(fid)
sizeEllipse = fscanf(fid, '%d', 1);
ellipses = [];
for i=1:sizeEllipse
    ellipses{i} = readEllipse(fid);
end
end

function ellipse = readEllipse(fid)
aux = fscanf(fid, '%f', [3 3]);
par = ellipse2param(aux);
% From C/C++ to matlab
par(1:2) = par(1:2) + 1;
ellipse = normalize_matr_det_1( param2ellipse(par) );
end

function flowComponents = readFlowComponents(fid, nCircles)

flowComponents = [ ];
sizeFlowComponents = fscanf(fid, '%d', 1);
for k=1:sizeFlowComponents
    % Read outer ellipse points
    sizeOuterEllipsePoints = fscanf(fid, '%d', 1);
    outerEllipsePoints = zeros(sizeOuterEllipsePoints,4);
    for i=1:sizeOuterEllipsePoints
        aux = fscanf(fid, '%d %d %d %d', [1 4]);
        aux(1:2) = aux(1:2) + 1;
        outerEllipsePoints(i,:) = aux;
    end
    flowComponent.outerEllipsePoints = outerEllipsePoints;
    
    % Read outer ellipse
    flowComponent.outerEllipse = readEllipse(fid);
    
    % Read filtered field lines
    sizeFilteredChildrens = fscanf(fid, '%d', 1);
    filteredFieldLines = zeros(sizeFilteredChildrens, nCircles, 4);
    for i=1:sizeFilteredChildrens
        for j=1:nCircles
            filteredFieldLines(i,j,1:4) = fscanf(fid, '%d %d %d %d', [1 4]);
            filteredFieldLines(i,j,1:2) = filteredFieldLines(i,j,1:2) + 1;
        end
    end
    flowComponent.filteredFieldLines = filteredFieldLines;
    
    % Read field lines
    sizeChildrens = fscanf(fid, '%d', 1);
    fieldLines = zeros(sizeChildrens, nCircles, 4);
    for i=1:sizeChildrens
        for j=1:nCircles
            fieldLines(i,j,1:4) = fscanf(fid, '%d %d %d %d', [1 4]);
            fieldLines(i,j,1:2) = fieldLines(i,j,1:2) + 1;
        end
    end
    flowComponent.fieldLines = fieldLines;
    
    % Read convex edge segment
    sizeConvexEdgeSegment = fscanf(fid, '%d', 1);
    convexEdgeSegment = zeros(sizeConvexEdgeSegment,4);
    for i=1:sizeConvexEdgeSegment
        aux = fscanf(fid, '%d %d %d %d', [1 4]);
        aux(1:2) = aux(1:2) + 1;
        convexEdgeSegment(i,:) = aux;
    end
    flowComponent.convexEdgeSegment = convexEdgeSegment;
    
    flowComponent.seed = fscanf(fid, '%d %d %d %d', [1 4]);
    
    flowComponents = [ flowComponents flowComponent ];
end
end

function points = readPoints(fid)
sizePointsVec = fscanf(fid, '%d', 1);
for i=1:sizePointsVec
    sizePoints = fscanf(fid, '%d', 1);
    if sizePoints == 0
        point = [];
    else
        for j=1:sizePoints
            aux = [ fscanf(fid, '%f %f', [1 2]) 1 ];
            aux(1:2) = aux(1:2) + 1;
            point(j,:) = aux;
        end
    end
    points{i} = point;
end
end

function idSet = readIdSet(fid)
sizeIdSet = fscanf(fid, '%d', 1);

idSet = [];
for i=1:sizeIdSet
    id = fscanf(fid, '%d',1);
    probaId = fscanf(fid, '%f',1);
    idSet = [ idSet ; [ id probaId ]];
end
end

function radiusRatio = readRadiusRatio(fid)
sizeRadiusRatios = fscanf(fid, '%d',1);

radiusRatio = [];
for i=1:sizeRadiusRatios
    ratio = fscanf(fid, '%f',1);
    radiusRatio = [ radiusRatio ; ratio ];
end
end

