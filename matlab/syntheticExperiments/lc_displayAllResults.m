% Input:
% iXp: experiment id:
%      iXp=1: vs. distance
%      iXp=2: vs. motion blur
%      iXp=3: vs. occlusion
% patterns:
%      patterns is a subset of {'3Crowns','4Crowns','ARTKPlus'},
%      For example: {'3Crowns'} or {'3Crowns','ARTKPlus'}.
% branchNames
%      branchNames is a couple of branches,
%      For example: {'cpu_adapt_of_gpu_part', 'optim_identify'}
%      Then the function will display the results of the detection obtained
%      for the synthetic dataset associated to the experiment iXp delivered
%      by each of these branches
% WARNING: if length(patterns) > 1, the results will be displayed comparing the
% results obtained for each patterns AND NOT for every branches. In that
% case, one branch has to be passed as input in branchNames.
function [] = lc_displayAllResults(iXp,patterns,branchNames)

disp('Read and evaluate the detection results');
allResultPath = lc_evaluateResultSynthCCTag(patterns, iXp, branchNames);

disp('Evaluation done');

disp('Display results');
lc_resultsAnalysis(patterns,iXp, branchNames, allResultPath);
disp('End');
