#Below are examples of how to run the BEES protocols on the sas_only test case presented in Bowerman et al. (In Preparation)

#Terminating the search after overfitting is observed
mpirun -np 6 BEES.py -o K63_sas_only -sas K63_saxs.dat -the_zip K63_sas_only_theoretical_profiles.zip -nMC 5 -iter 10000 -burn 1000 -dmax 83.6

#Building models for all possible sub-basis combinations
mpirun -np 6 BEES.py -o K63_sas_only_every -sas K63_saxs.dat -the_zip K63_sas_only_theoretical_profiles.zip -nMC 5 -iter 10000 -burn 1000 -dmax 83.6 --every
