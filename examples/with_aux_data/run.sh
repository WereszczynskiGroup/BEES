#Below are examples of how to run the BEES protocols on the auxiliary data-included test case presented in Bowerman et al. (In Preparation)

#Terminating the search after overfitting is observed
mpirun -np 6 BEES.py -o K63_with_aux -sas K63_saxs.dat -aux K63_dist_and_angle.dat -the_zip K63_with_auxiliary_theoretical_profiles.zip -nMC 5 -iter 10000 -burn 1000 -dmax 83.6

#Building models for all possible sub-basis combinations
mpirun -np 6 BEES.py -o K63_with_aux_every -sas K63_saxs.dat -aux K63_dist_and_angle.dat -the_zip K63_with_auxiliary_theoretical_profiles.zip -nMC 5 -iter 10000 -burn 1000 -dmax 83.6 --every
