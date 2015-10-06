function fn = lc_loadMarkers(nCrowns, idXp, branchName)

addpath('..');

fn = lc_syntheticFileNames(nCrowns,idXp,branchName);
fn

listing = dir(fn.folderName);

listing = listing(find(cellfun(@(x) x == 1 ,{listing(:).isdir})));

if 0 % Degub : disp list of contained files
    for i=1:length(listing)
        listing(i)
    end
end

nFrames = length(listing)-5;

% '.', '..', localization, identification and resultForMatlab are not considered.
for i=3:nFrames+2
    file{i-2} = listing(i).name;
end
nFrames
if (nFrames <= 0)
   error('CCTag error: Unable to find any detection result to load'); 
end
file = sort(file);

display_ = 0;

% Read each files to build each markers in each views.
for i = 1:nFrames
    imagePath = [ fn.folderName '/' file{i} ];
    nameFile = [ imagePath '/identification.txt' ];
    if ( strcmp(nCrowns, '3Crowns') || strcmp(nCrowns, '4Crowns') )
        Markers{i} = parseCCTags(nameFile, imagePath);
    elseif strcmp(nCrowns, 'ARTKPlus')
        imageFileName = [ fn.filePictures '/' file{i} '.png' ];
        Markers{i} = parseARTKPlus(nameFile, imageFileName);
    end
end

folderPath = [ fn.folderName '/resultForMatlab' ];

system( [ 'mkdir -p ' folderPath ] );

fn.rawDetectionResultPath = [ folderPath '/rawDetectionResult.mat' ];
fn.statisticalEvalResultPath = [ folderPath '/statisticalEvalResult.mat' ];

save( fn.rawDetectionResultPath ,'Markers');

end
