% For the documentation, cf. the one in lc_displayAllResults.m

iXp = 3; % in {1,2,3}
patterns = { '4Crowns' };% Subset of {'3Crowns','4Crowns','ARTKPlus'}.
                             % Only one must be passed for branches
                             % comparison.
branchNames = {'cpu_adapt_of_gpu_part', 'optim_identify'};
                             % It can be any of the available branches.
                             % If only one pattern is passed as input, then
                             % a couple of branches can be passed for
                             % comparison.

lc_displayAllResults(iXp,patterns,branchNames);