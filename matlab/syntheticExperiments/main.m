% For the documentation, cf. the one in lc_displayAllResults.m

setPath;

iXp = 1; % in {1,2,3}
patterns = { '3Crowns' };% Subset of {'3Crowns','4Crowns','ARTKPlus'}.
                             % Only one must be passed for branches
                             % comparison.
branchNames = {'cpu_adapt_of_gpu_part', 'optim_identify'};
                             % It can be any of the available branches.
                             % If only one pattern is passed as input, then
                             % a couple of branches (a maximum of 2) can be passed for
                             % comparison.

lc_displayAllResults(iXp,patterns,branchNames);
