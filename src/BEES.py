#!/usr/bin/env python
'''
    BEES is a module that will read in a list of scattering
    profiles, along with an optional second set of measurements, and it will
    reweight populations in subsets of these scattering profiles to find
    the best fit to the experimental data. Here, "best fit" means a balance
    between a model with a low chi^2 value that simultaneously uses as few
    fitting parameters (number of profiles) as possible, thereby trying to
    avoid overfitting. This module will output information regarding this
    "best model", as well as provide graphical and quantitative means to
    compare the performance of all other sampled models. If you use the BEES
    program please be sure to cite the references below.

    REFERENCE:

    Bowerman et al.
    Journal of Chemical Theory and Computation, 13, pp 2418-2429 (2017)

    Bowerman et al.
    In Preparation
'''
import numpy as np
import mpi4py
from mpi4py import MPI
import os
import io
import sys
import zipfile
import itertools
import time
import pickle
import subprocess
import bokeh
import argparse
import bayesian_ensemble_estimator_parallel_routine as BEES_parallel

app = 'bayesian_ensemble_estimator'

class module_variables():
    def __init__(self, parent=None):
        self.app = app

class efvars():
    ''' _E_nsemble _F_it _VAR_iable_S_ '''
    def __init__(self, parent=None):
        pass

######## MPI Environment ########
comm=MPI.COMM_WORLD
rank=comm.Get_rank()
size=comm.Get_size()
#################################


# This is the top-level object for managing the Bayesian Ensemble Fit Routine
'''
    This object builds up all the necessary variables and passes them to the
    bayesian_ensemble_estimator_parallel_routine object that runs the
    reweighting protocol.
'''


class ensemble_routine(object):
    def __init__(self, parent=None,cmd_line_args=None):
        self.mvars  = cmd_line_args
        mvars       = self.mvars

        mvars.nproc = size
        if mvars.d_max > 0.0:
            mvars.shansamp = True
        else:
            mvars.shansamp = False

        

    def main(self):
        self.efvars = efvars()
        self.Initialize()
        if rank==0:
            self.PickleVars()
        if rank==0:
            print("\n%s \n" % ('=' * 60))
            submit_time = time.asctime(time.gmtime(time.time()))
            print("DATA FROM RUN: %s \n\n" % submit_time)
            print('STATUS\t0.0001\n\n')
        self.EnsembleFit()
        if rank==0:
            print(efvars.best_model)
            self.epilogue()


    def Initialize(self):
        '''
        Prepare efvars object
        '''
        mvars = self.mvars
        # Double-check that the user isn't asking for multiple IC's.
        # Default according to array index, if so.
        ic_list = np.array([mvars.use_bic,mvars.use_aic,mvars.use_dic,
                            mvars.use_waic2,mvars.use_waic1],dtype=bool)
        flagged = np.where(ic_list)[0]
        #If user did not specify, use BIC
        if len(flagged) == 0:
            mvars.use_bic = True
        elif len(flagged) > 1:
            mvars.use_bic   = False
            mvars.use_aic   = False
            mvars.use_dic   = False
            mvars.use_waic2 = False
            mvars.use_waic1 = False

            which_ic = flagged[0]
            if which_ic==0:
                mvars.use_bic   = True
            elif which_ic==1:
                mvars.use_aic   = True
            elif which_ic==2:
                mvars.use_dic   = True
            elif which_ic==3:
                mvars.use_waic2 = True
            elif which_ic==4:
                mvars.use_waic1 = True

        efvars = self.efvars
        if rank == 0:
            
            efvars.output_folder = os.path.join(mvars.runname, app)
            efvars.pickle_folder = os.path.join(efvars.output_folder,
                                            'pickle_files')
            efvars.log_folder = os.path.join(efvars.output_folder,
                                            'logfiles')
            if not os.path.isdir(mvars.runname):
                os.mkdir(mvars.runname)
            if not os.path.isdir(efvars.output_folder):
                os.mkdir(efvars.output_folder)
            if not os.path.isdir(efvars.pickle_folder):
                os.mkdir(efvars.pickle_folder)
            if not os.path.isdir(efvars.log_folder):
                os.mkdir(efvars.log_folder)
            # Clean out old log files, backs-up most recent version
            existingfiles = os.listdir(efvars.output_folder)
            for item in existingfiles:
                if item.endswith('BayesMC.log'):
                    oldname = os.path.join(efvars.output_folder, item)
                    newname = os.path.join(efvars.output_folder, item+'.BAK')
                    os.rename(oldname, newname)

            try:
                efvars.Q, efvars.I, efvars.ERR = np.genfromtxt(mvars.sas_data,
                                                           unpack=True,
                                                           usecols=(0, 1, 2))
            except:
                print('ERROR: Unable to load interpolated data file: '
                      + mvars.sas_data)

            # Set up status file for progress tracking
            efvars.status_file = os.path.join(efvars.output_folder,
                                          '._status.txt')
            statf = open(efvars.status_file, 'w')
            statf.write('STATUS\t0.0001\n')
            statf.close()

            if mvars.shansamp:
                self.BuildShannonSamples()
                efvars.samples_Q = efvars.Q[efvars.shannon_samples]
                efvars.samples_I = efvars.I[efvars.shannon_samples]
                efvars.samples_ERR = efvars.ERR[efvars.shannon_samples]
                efvars.num_q = efvars.number_of_channels
            else:
                efvars.samples_Q = efvars.Q
                efvars.samples_I = efvars.I
                efvars.samples_ERR = efvars.ERR
                if mvars.num_q > 0:
                    efvars.num_q = mvars.num_q
                else:
                    efvars.num_q = len(efvars.Q)

            if mvars.auxiliary_data != '':
                efvars.include_second_dimension = True
                try:
                    efvars.aux_data, efvars.aux_error = np.genfromtxt(
                        mvars.auxiliary_data, usecols=(0, 1),
                        unpack=True, dtype=float)
                    efvars.num_aux = len(efvars.aux_data)
                except:
                    print('ERROR: Unable to load auxiliary experimental'
                          + ' data file: '+mvars.auxiliary_data)
            else:
                efvars.include_second_dimension = False
                efvars.num_aux = 0

            efvars.num_points = efvars.num_q + efvars.num_aux
            self.UnpackTheoreticalProfiles()
            self.BuildBasis()

            for thread in range(1,size):
                comm.send(efvars,dest=thread)
        else:
            self.efvars = comm.recv(source=0)
        return

    def BuildShannonSamples(self):
        '''
        Build Shannon Channels for Calculating Chi_free
        '''
        mvars = self.mvars
        efvars = self.efvars

        shannon_width = np.pi/mvars.d_max
        n_channels = int(np.floor(efvars.Q[-1]/shannon_width))

        shannon_points = np.random.choice(np.where(
            np.logical_and(efvars.Q >= 0,
                           efvars.Q < shannon_width))[0], 3001)
        for idx in range(1, n_channels):
            this_channel = np.random.choice(np.where(
                np.logical_and(efvars.Q >= idx * shannon_width,
                               efvars.Q < (idx+1) * shannon_width))[0], 3001)
            shannon_points = np.vstack((shannon_points, this_channel))

        efvars.shannon_samples = np.copy(shannon_points)
        efvars.number_of_channels = n_channels
        return

    def UnpackTheoreticalProfiles(self):
        mvars = self.mvars
        efvars = self.efvars

        efvars.profiles_directory = os.path.join(efvars.output_folder,
                                                 'TheoreticalProfiles')
        if os.path.isdir(efvars.profiles_directory):
            os.system('rm -r '+efvars.profiles_directory)
        if mvars.debug:
            print('efvars.profiles_directory - '+ efvars.profiles_directory)
        archive = zipfile.ZipFile(mvars.theoretical_profiles_zip)
        archive.extractall(efvars.profiles_directory)
        archive.close()

        return

    def BuildBasis(self):
        '''
        Stores individual theoretical profiles into single array(s) for
        faster ensemble averaging.
        '''
        mvars = self.mvars
        efvars = self.efvars

        flist_as_str = os.path.join(efvars.profiles_directory, 'filelist.txt')
        efvars.name_array = np.genfromtxt(flist_as_str, usecols=0, dtype=str)
        efvars.number_of_profiles = len(efvars.name_array)

        # Below is used to quickly build subsets, even if it seems redundant
        efvars.id_array = np.arange(efvars.number_of_profiles, dtype=int)
        efvars.subspace_dict = {}
        efvars.full_scattering_basis = np.zeros((efvars.number_of_profiles,
                                                 len(efvars.Q)), dtype=float)

        if efvars.include_second_dimension:
            try:
                efvars.aux_name_array = np.genfromtxt(flist_as_str,
                                                      usecols=1, dtype=str)
                efvars.number_of_aux_profiles = len(efvars.aux_name_array)
                if efvars.number_of_aux_profiles != efvars.number_of_profiles:
                    print('ERROR: A different number of extra profiles'
                              + ' ('+str(efvars.number_of_aux_profiles)+')'
                              + ' have been supplied than the number of'
                              + ' scattering profiles ('
                              + str(efvars.number_of_profiles)+').')
                size_tuple = (efvars.number_of_profiles, efvars.num_aux)
                efvars.full_extra_basis = np.zeros(size_tuple, dtype=float)
            except:
                print('ERROR: In order to use a second dataset,'
                          + ' users must include the list of files in'
                          + ' the second column of the'
                          + ' \"filelist.txt\" file.\"')
        else:
            efvars.full_extra_basis = None

        for idx in efvars.id_array:
            try:
                scatter_file = os.path.join(efvars.profiles_directory,
                                            efvars.name_array[idx])
                the_q, the_i = np.genfromtxt(scatter_file, dtype=float,
                                             usecols=(0, 1), unpack=True)

                if not np.all(the_q == efvars.Q):
                    print('ERROR: Q-values of '+scatter_file
                              + ' don\'t match the experimental values.')
                efvars.full_scattering_basis[idx] = the_i
            except:
                print('ERROR: Data file ('+scatter_file+')'
                          + ' not found. Please inspect the .zip archive'
                          + ' and re-upload.')

            if efvars.include_second_dimension:
                extra_file = os.path.join(efvars.profiles_directory,
                                          efvars.aux_name_array[idx])
                try:
                    extra_data = np.genfromtxt(extra_file, usecols=0,
                                               dtype=float)
                except:
                    print('ERROR: Data file ('
                              + ' '+extra_file+') not found.'
                              + ' Please inspect the .zip archive'
                              + ' and re-upload.')
                if len(extra_data) != efvars.num_aux:
                    print('ERROR: The number of points in '
                              + extra_file + ' does not match'
                              + ' the number of experimental values'
                              + ' ('+str(efvars.num_aux)+').')
                efvars.full_extra_basis[idx] = extra_data

    def PickleVars(self):
        '''
        Saves self.mvars and self.efvars to serialized files that are
        loaded by the parallel runtime
        '''
        mvars = self.mvars
        efvars = self.efvars
        if mvars.debug:
            print('In PickleVars.')

        self.mvarspickle = os.path.join(
            efvars.pickle_folder, mvars.runname+'_mvars.p')
        pickle.dump(mvars, open(self.mvarspickle, 'wb'))
        if mvars.debug:
            print('mvars have been written to pickle: '+self.mvarspickle)

        self.efvarspickle = os.path.join(
            efvars.pickle_folder, mvars.runname+'_efvars.p')
        pickle.dump(efvars, open(self.efvarspickle, 'wb'))
        if mvars.debug:
            print('efvars have been written to pickle: '+self.efvarspickle)

        return

    def EnsembleFit(self):
        '''
        This initializes the parallel iterative fitting routine
        '''
        mvars = self.mvars
        efvars = self.efvars

        BEES_routine = BEES_parallel.ensemble_routine()
        BEES_routine.main(mvars=mvars,efvars=efvars)

    def epilogue(self):

        efvars = self.efvars
        mvars = self.mvars

        try:
            print('Best model found: \n\n')
            parallel_outf = os.path.join(efvars.output_folder,
                                         mvars.runname+'_final_model.dat')
            sas_outf = os.path.join(efvars.output_folder,
                                    mvars.runname+'_ensemble_sas.int')
            all_model_outf = os.path.join(efvars.output_folder,
                                          mvars.runname+'_all_models.dat')
            plots_outf = os.path.join(efvars.output_folder,
                                      mvars.runname+'_plots.html')
            model_dataf = open(parallel_outf, 'r')
            for line in model_dataf:
                print(line)
                print('\n\n')
            model_dataf.close()

        except:
            print('ERROR: Unable to locate parallel routine output'
                      + ' (parallel routine may have crashed)!')

        print('\nModel populations, ensemble spectra, and interactive plot HTML files saved in'
             + ' '+efvars.output_folder+'\n\n')
        os.remove(efvars.status_file)
        print('\nBest model spectra plotted below.\n\n')
        print('STATUS\t1.0\n\n')
        print("\n%s \n" % ('=' * 60))



cmd_parser  = argparse.ArgumentParser()
cmd_parser.add_argument("-o",dest='runname',type=str,help="Prefix for file and subdirectory generation (Default = \"BEES_run\")",default="BEES_run")
cmd_parser.add_argument("-sas",dest="sas_data",type=str,help="Location of the experimental SAS file to which the ensemble will be fit.",required=True)
cmd_parser.add_argument("-the_zip",dest="theoretical_profiles_zip",type=str,help="Location of the theoretical candidate states zip archive",required=True)
cmd_parser.add_argument("-nMC",dest='number_of_MCs',type=int,help='Number of MC replicas to run per sub-basis set.  (Default = 5)',default=5)
cmd_parser.add_argument("-iter",dest='max_iterations',type=int,help='Number of iterations to run for each MC replica. (Default = 10000)',default=10000)
cmd_parser.add_argument("-burn",dest="posterior_burn",type=int,help='Number of steps to remove from the beginning of each MC trajectory; removes influence of initial random populations from final estimation. (Default = 1000)',default=1000)
cmd_parser.add_argument("-dmax",dest="d_max",type=float,help='Experimentally-derived D_max value from the SAS profile; used to conduct Shannon sampling.', default=0.0)
cmd_parser.add_argument("-aux",dest="auxiliary_data",type=str,help='Location of the auxiliary data that will be fit simultaneously with the SAS spectrum.',default='')
cmd_parser.add_argument("-numQ",dest='num_q',type=int,help='User-defined number of independent scattering points.  Value is over-ridden when "-dmax" is non-zero',default=0)
cmd_parser.add_argument("--aic",dest='use_aic',action='store_true',help='Use AIC to evaluate overfitting.',default=False)
cmd_parser.add_argument("--bic",dest='use_bic',action='store_true',help='Use BIC to evaluate overfitting. (Recommended)',default=False)
cmd_parser.add_argument("--dic",dest='use_dic',action='store_true',help='Use DIC to evaluate overfitting.',default=False)
cmd_parser.add_argument("--waic1",dest='use_waic1',action='store_true',help='Use WAIC to evaluate overfitting, using P1 estimate for number of free parameters (see documentation).',default=False)
cmd_parser.add_argument("--waic2",dest='use_waic2',action='store_true',help='Use WAIC to evaluate overfitting, using P2 estimate for number of free parameters (see documentation).',default=False)
cmd_parser.add_argument("--use_all",dest="use_all",action='store_true',help='Fit only the individual candidates and the collection of all candidates (do not conduct iterative scan for overfitting)',default=False)
cmd_parser.add_argument("--every",dest="every",action='store_true',help='Model populations for every combination of sub-basis sizes (conduct iterative fitting, but ignore overfitting \"STOP\" signal)',default=False)
cmd_parser.add_argument("-sigma",dest='sigma',type=float,help='Width of the Gaussian distribution used to weighted-random select population iterations. (Default = 0.10)',default=0.10)
cmd_parser.add_argument("-thresh",dest='zeroing_threshold',type=float,help='Populations below this value at each iteration of the Bayesian MC search will be set to 0.0, and the weights will be re-normalized. (Default = 0.0)',default=0.0)
cmd_parser.add_argument("--walk_one",dest='walk_one',action='store_true',help='Increment one population at a time, instead of randomly incrementing every population at each iteration',default=False)
cmd_parser.add_argument("--v",dest="debug",action='store_true',default=False,help="Run in debug (high verbosity) mode.")
cmd_args    = cmd_parser.parse_args()

reweighting = ensemble_routine(cmd_line_args=cmd_args)
reweighting.main()
