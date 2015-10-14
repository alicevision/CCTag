function lc_generateViews(idXp,doWrite)

if nargin < 2
    doWrite = 1;
end

setPath;

sTypeMarkers = { 'ARTKPlus' , '3Crowns', '4Crowns' };

lc_generateSynthCCTag(sTypeMarkers, idXp, doWrite);

'Imaged CCTags generated'