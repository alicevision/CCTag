function allResultPath = lc_evaluateResultSynthCCTag(sTypeMarkers, idXp, branchNames)

close all

for iCrown = 1:length(sTypeMarkers)%nCrowns = sTypeMarkers
    for iBranch = 1:length(branchNames)%for branchName = branchNames
        
        nCrowns = sTypeMarkers{iCrown};
        branchName = branchNames{iBranch};
        
        % Maximum number of treated marker
        if strcmp(nCrowns, '3Crowns')
            nMarker = 32;
        elseif strcmp(nCrowns, '4Crowns')
            nMarker = 294;
        elseif strcmp(nCrowns,'ARTKPlus')
            nMarker = 512;
        end
        
        % Load experience parameters
        xp = lc_loadVaryingParameters(idXp);
        nTest = xp.nTest;
        distance = xp.distance;
        angle = xp.angle;
        focale = xp.focale;
        sigmaNoise = xp.sigmaNoise;
        lengthMotionBlur = xp.lengthMotionBlur;
        occlusion = xp.occlusion;
        
        % Parse and load detection results
        fn = lc_loadMarkers( nCrowns, idXp, branchName);
        load( fn.rawDetectionResultPath, 'Markers');
        
        allResultPath{iCrown,iBranch} = fn;
        
        % Initialize the index
        ind = 1;
        
        % Initialize results
        nbConfusion = zeros(length(distance), length(angle), length(focale), length(sigmaNoise), length(lengthMotionBlur),length(occlusion));
        nbNegatifs  = zeros(length(distance), length(angle), length(focale), length(sigmaNoise), length(lengthMotionBlur),length(occlusion));
        precision  = zeros(length(distance), length(angle), length(focale), length(sigmaNoise), length(lengthMotionBlur),length(occlusion));
        
        % Set path to the ground truth data
        syntheticDataPath = [ DATA_PATH '/cctag/bench/images/synthetic' ];
        xpPath = [syntheticDataPath '/' nCrowns '/xp' int2str(idXp)];
        
        for iDistance = 1:length(distance)
            for iAngle = 1:length(angle)
                for iFocale = 1:length(focale)
                    for iSigmaNoise = 1:length(sigmaNoise)
                        for iLengthMotionBlur = 1:length(lengthMotionBlur)
                            for iOcclusion = 1:length(occlusion)
                                
                                nPrec = 0;
                                
                                for nt = 1:nTest
                                    
                                    %'distance(iDistance), angle(iAngle), focale(iFocale), sigmaNoise(iSigmaNoise), lengthMotionBlur(iLengthMotionBlur), occlusion(iOcclusion), nt)'
                                    
                                    data_path = sprintf('%6.4f_%6.4f_%6.4f_%6.4f_%6.4f_%6.4f_%d', ...
                                        distance(iDistance), angle(iAngle), focale(iFocale), sigmaNoise(iSigmaNoise), lengthMotionBlur(iLengthMotionBlur), occlusion(iOcclusion), nt);
                                    
                                    % Same as in lc_generateSynthCCTag.m
                                    aux = load(sprintf('%s/groundTruth/%s.mat', xpPath, data_path));
                                    
                                    markers = Markers{aux.ind};
                                    
                                    % Check if the marker is detected or not
                                    if isempty(markers)
                                        nbNegatifs(iDistance,iAngle,iFocale, iSigmaNoise, iLengthMotionBlur,iOcclusion) = nbNegatifs(iDistance,iAngle,iFocale, iSigmaNoise,iLengthMotionBlur,iOcclusion) + 1;
                                        %nbConfusion(iDistance,iAngle,iFocale, iSigmaNoise, iLengthMotionBlur,iOcclusion) = nbConfusion(iDistance,iAngle,iFocale, iSigmaNoise,iLengthMotionBlur,iOcclusion) + 1;
                                        continue;
                                    end
                                    
                                    markerRes = markers(1);
                                    
                                    % Check if the marker is correctly identified
                                    if (markerRes.id ~= aux.id)
                                        nbConfusion(iDistance,iAngle,iFocale, iSigmaNoise, iLengthMotionBlur,iOcclusion) = nbConfusion(iDistance,iAngle,iFocale, iSigmaNoise,iLengthMotionBlur,iOcclusion) + 1;
                                        %nbConfusion(iAngle,iDistance) = nbConfusion(iAngle,iDistance) + 1;
                                    else
                                        
                                        if strcmp(nCrowns, '4Crowns') || strcmp(nCrowns, '3Crowns')
                                            precision(iDistance,iAngle,iFocale,iSigmaNoise, iLengthMotionBlur,iOcclusion) = ...
                                                ( precision(iDistance,iAngle,iFocale,iSigmaNoise, iLengthMotionBlur,iOcclusion)*nPrec + ...
                                                norm(markerRes.imgCenter - normalize(aux.metricHomography*[0;0;1])) )/(nPrec+1);
                                        elseif strcmp(nCrowns,'ARTKPlus')
                                            % Compute the intersection of the the v1-v3
                                            % and v2-v4 segments
                                            imCenter = cross(...
                                                cross(markerRes.vertex(:,1),markerRes.vertex(:,3)),...
                                                cross(markerRes.vertex(:,2),markerRes.vertex(:,4)));
                                            imgCenter =  normalize(imCenter);
                                            
                                            precision(iDistance,iAngle,iFocale,iSigmaNoise, iLengthMotionBlur,iOcclusion) = ...
                                                ( precision(iDistance,iAngle,iFocale,iSigmaNoise, iLengthMotionBlur,iOcclusion)*nPrec + ...
                                                norm(imgCenter - normalize(aux.metricHomography*[0;0;1])) )/(nPrec+1);
                                        end
                                        nPrec = nPrec+1;
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
        
        nbNegatifs = squeeze(nbNegatifs);
        nbConfusion = squeeze(nbConfusion);
        precision = squeeze(precision);
        
        % Save the result
        save( fn.statisticalEvalResultPath );
        
    end
end
