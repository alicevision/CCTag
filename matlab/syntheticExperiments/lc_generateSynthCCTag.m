function lc_generateSynthCCTag(sTypeMarkers, iXp, doWrite)

if nargin < 3
    doWrite = 0;
end

close all

allLibs = [];

syntheticDataPath = [ DATA_PATH '/cctag/bench/images/synthetic' ];

for nCrowns = sTypeMarkers
    xpPath = [syntheticDataPath '/' nCrowns{1} '/xp' int2str(iXp)]
    % Set all path
    nLib.imageToTestPath = [ xpPath '/images' ]
    system( [ 'mkdir -p ' nLib.imageToTestPath ] );
    if strcmp(nCrowns, '3Crowns')
        nLib.pngLibrary = [ MTWP_PATH '/cctag/generatorThreeCrowns/cctag_png'];
        % Maximum number of treated marker
        nLib.nMarker = 32;
        nLib.typePrimitive = 'circular';
        nLib.typeMarker = nCrowns;
    elseif strcmp(nCrowns, '4Crowns')
        nLib.pngLibrary = [ MTWP_PATH '/cctag/generatorFourCrowns/cctag_png'];
        % Maximum number of treated marker
        nLib.nMarker = 40;%128;
        nLib.typePrimitive = 'circular';
        nLib.typeMarker = nCrowns;
    elseif strcmp(nCrowns,'ARTKPlus')
        nLib.pngLibrary = [ MTWP_PATH '/cctag/syntheticExperiments/ARTKPlusId' ];
        % Maximum number of treated marker
        nLib.nMarker = 40;%512;
        nLib.typePrimitive = 'square';
        nLib.typeMarker = nCrowns;
    else
        error('Unknown marker type');
    end
    
    % Get image files in the marker path
    files = dir(nLib.pngLibrary);
    
    sizeUp = 1100;
    I = zeros(sizeUp,sizeUp,nLib.nMarker);
    margin = round((sizeUp-800)/2);
    
    % Load each marker image
    for iImage=1:min(nLib.nMarker,length(files)-2)
        imagePath = [ nLib.pngLibrary '/' files(iImage+2).name ];
        aux = double(imread(imagePath));
        vmax = max(max(aux));
        
        aux2 = aux';
        aux = ones(sizeUp,sizeUp)*vmax;
        aux(margin+1:end-margin,margin+1:end-margin) = aux2;
        
        I(:,:,iImage) = uint8(aux*255/vmax);
    end
    nLib.I = I;
    allLibs = [ allLibs nLib ];
end

% Reference index
ind = 1;

xp = lc_loadVaryingParameters(iXp);
nTest = xp.nTest;
distance = xp.distance;
fixedDistance = xp.fixedDistance;
angle = xp.angle;
fixedAngle = xp.fixedAngle;
focale = xp.focale;
focalRange = xp.focalRange;
sigmaNoise = xp.sigmaNoise;
lengthMotionBlur = xp.lengthMotionBlur;
occlusion = xp.occlusion;

for iDistance = 1:length(distance)
    for iAngle = 1:length(angle)
        for iFocale = 1:length(focale)
            for iSigmaNoise = 1:length(sigmaNoise)
                for iLengthMotionBlur = 1:length(lengthMotionBlur)
                    for iOcclusion = 1:length(occlusion)
                        
                        for nt = 1:nTest
   
                            vPatterns = [];
                            
                            % Loop over all the types of marker
                            for nLib = allLibs
                                id = round(0.5+nLib.nMarker*rand);
                                pattern.id = id;
                                pattern.typePrimitive = nLib.typePrimitive;
                                pattern.typeMarker = nLib.typeMarker;
                                pattern.I = nLib.I(:,:,id);
                                pattern.imageToTestPath = nLib.imageToTestPath;
                                vPatterns = [ vPatterns pattern ];
                            end
                            
                            % Generate random homography with respect to the
                            % camera pose define by all the following parameters
                            % and the wrapped image is then corrupted by
                            % some noise (optional)
                            [ vImagedPatterns, homography, metricHomography] = lc_generateRandomHomographie( vPatterns, distance(iDistance), fixedDistance, ...
                                angle(iAngle), fixedAngle, focale(iFocale), focalRange, sigmaNoise(iSigmaNoise), lengthMotionBlur(iLengthMotionBlur), occlusion(iOcclusion) );
                            
                            data_path = sprintf('%6.4f_%6.4f_%6.4f_%6.4f_%6.4f_%6.4f_%d', ...
                                distance(iDistance), angle(iAngle), focale(iFocale), sigmaNoise(iSigmaNoise), lengthMotionBlur(iLengthMotionBlur), occlusion(iOcclusion), nt);
                            
                            % save information about the picture to evaluate
                            % detection results
                            
                            indInMarker = ind;
                            
                            if doWrite
                                for nImagedPattern = 1:length(vImagedPatterns)
                                    
                                    imPattern = vImagedPatterns(nImagedPattern)
                                    nCrowns = imPattern.typeMarker
                                    
                                    id = imPattern.id;
                                    
                                    % Path where to save the ground truth data
                                    xpPath = [syntheticDataPath '/' nCrowns{1} '/xp' int2str(iXp)];
                                    relativeFolderName = sprintf('%s/groundTruth',xpPath);
                                    system( [ 'mkdir -p ' relativeFolderName ] );
                                    save(sprintf('%s/%s.mat',relativeFolderName,data_path), 'ind','nt','id','distance', 'angle', 'focale', 'sigmaNoise', 'lengthMotionBlur', 'occlusion', 'homography', 'metricHomography', 'indInMarker');

                                    % Path where to save the generated image
                                    sprintf('%s/%05d.png',imPattern.imageToTestPath, ind)
                                    imwrite(imPattern.I, sprintf('%s/images/%05d.png',xpPath, ind), 'png');
                                end
                            end
                            
                            ind = ind+1;
                        end
                    end
                end
            end
        end
    end
end


%   data_path = sprintf(['mat/%d/%d/allData_nt_%d_nFrame_%d_nPlane_%d_distor_%f_sizeMarker_%f' ...
%  '_fRef_%f_sigmaPP_%f_upFixed_%d_planeRotate_%d_sigma_%f' ...
% '_nbPtsGrid_%d_sigmaTrans_%f_distanceToCentre_%f_missingPlane_%d_sigmaF_%f_nNaturalPoint_%d.mat'],...

%        save(data_path,'animGTorig','animGT','adjustedHomographies',...
%       'nt','nFrame','nPlane','kc','sizeMarker',...
%       'fRef','sigmaPP','anim','upFixed',...
%       'planeRotate','sigma','nbPtsGrid','sigmaTrans',...
%       'distanceToCentre','missingPlane', 'Hres');
