# Installation Instructions for BEES

 1. [Download the appropriate Anaconda Python2 distribution](https://www.anaconda.com/download/)
 
 2. After Anaconda is installed, use the `conda` command to download the required Python libraries to run BEES:
    `conda install numpy=1.13.1 scipy=0.19.1 bokeh=0.12.16 mpi4py=2.0.0`
    
    * You may be interested in separating the BEES Anaconda installation from their other Anaconda environments so as to avoid overwriting .  If so, you should first use the following command to create a separate Anaconda virtual environment for BEES before installing the required libraries:
    `conda create -n BEESenv anaconda`
    * The BEES Anaconda environment can now be loaded using the following command:
    `source activate BEESenv`
    
 3. Download the `BEES.py` and `bayesian_ensemble_estimator_parallel_routine.py` files and ensure that they are in your local `PATH` and `PYTHONPATH` environment variables.
 
    * You should also ensure that the `BEES.py` file is executable!  On Mac and Linux machines, this can be accomplished by navigating to the folder containing `BEES.py` and then calling: `chmod +x BEES.py`
   
 4. You can now run BEES.py from your command line!
