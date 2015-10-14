function fn = lc_syntheticFileNames(nCrowns,idXp,branchName)

syntheticDataPath = [ DATA_PATH '/cctag/bench/images/synthetic' ];

fn.filePictures = [ syntheticDataPath '/' nCrowns '/xp' int2str(idXp) '/images' ];

if strcmp(nCrowns, '3Crowns')
    % List all files in folder name that contains all markers info. One file is associated to one view (of a video).
    fn.folderName = [ syntheticDataPath '/' nCrowns '/xp' int2str(idXp) '/images/' branchName '/cctag3CC' ];
elseif strcmp(nCrowns, '4Crowns')
    % List all files in folder name that contains all markers info. One file is associated to one view (of a video).
    fn.folderName = [ syntheticDataPath '/' nCrowns '/xp' int2str(idXp) '/images/' branchName '/cctag4CC' ];
elseif strcmp(nCrowns, 'ARTKPlus')
    fn.folderName = [ syntheticDataPath '/' nCrowns '/xp' int2str(idXp) '/images/' branchName '/artoolkit' ];
end
