# Bayesian Ensemble Estimator from SAS (BEES)

BEES is a python program designed to fit ensemble models to experimental SAS by drawing combinations of states from a candidate pool of structures.  This is done using a Bayesian Monte Carlo (BMC) parameter search routine that iteratively increases the number of states per combination, so as to avoid overfitting the data with an arbitrary number of fitting parameters (number of populations).

## Basic Input Parameters

In order to run BEES, you must have two required files: an empirical scattering curve (3-column file: column 1 = q, column 2 = I(q), column 3 = error) and a ZIP archive containing the scattering curves for each of the candidate populations.  Additionally, users may also include auxiliary data (2-column file: column 1 = measurement, column 2 = error in measurement). 

## Example Work Flow

An example for how BEES might fit in a SAS modeling workflow is as follows:

  1.  Gather empirical SAS data

      1b.  Post-process the SAS data to filter spurious data, such as removing low-q beam smearing effects and truncating the high-q region.
      
      1c.  Calculate the D<sub>max<\sub> value of your molecular system from the empirical data.
  
      1d.  (Optional) Gather an additional set of empirical measurements to use as auxiliary data in the BEES fitting algorithm.
 
  2.  Conduct Monte Carlo or Molecular Dynamics simulations of your molecule of interest.
  
  3.  Cluster the structural states to a managable size of candidates (N ~ 15, for example).
  
  4.  Calculate the scattering profile of each state using your favorite theoretical scattering calculator.
    
      4b. (Optional) Calculate theoretical profiles of each state for the set of auxiliary measurements
      
      4c.  Create a `filelist.txt` file that contains a list of all the scattering (and auxiliary, if determined) profiles that you wish to consider for population re-weighting.
      
      4d.  ZIP `filelist.txt` and all your theoretical profiles (including theoretical auxiliary data) into a single ZIP archive.
      
  5. Use these files as inputs to the BEES.py program.
  
  6. Analyze the performance of both the best model and other competing models using the text and graphical HTML files created by the BEES.py program.
