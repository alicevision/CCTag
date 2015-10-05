In order to run the tests on the synthetic data located on the CCTagDataset repository, 
you have to follow the following instructions:
1. Create/use your favourite 'data' folder (e.g. $HOME/data)
2. cd path_to_your_data_folder 
3. git clone git@github.com:poparteu/CCTagDataset.git
4. mv CCTagDataset cctag
5. Edit you .bashrc and add the following environment variable:
   export DATA_PATH=path_to_your_data_folder (e.g. $HOME/data)
6. Open a new terminal
7. Install octave if not already installed
8. Go to ./matlab/syntheticExperiments/octaveVersion
9. Start octave
10. Run the cctag detection on every branches you want to test via
    ./script/runSynthetic.sh nbRings idXp branchName
10. Edit main.m to set which experiments you want to run, for which pattern and on which branch(es)
11. Save and launch main.m in the octave terminal: main
