import sys
import os
import time
import argparse
from datetime import datetime, timedelta
import numpy as np
import torch
from torch.utils import data
import pickle

# Define your path to the repo. 
path_for_your_repo = "/Users/ozanbaris/Documents/GitHub/IbexRLExperiments/BoptestExperiment"

BOPTEST_GYM_PATH = f'{path_for_your_repo}/Gnu-RL/project1-boptest-gym' # Path to boptest-gym directory
repo_path = os.path.abspath(f"{path_for_your_repo}/Ibex_RL/agent")  # Adjust this path as necessary to locate your repo's root
gnu_rl_path = f"{path_for_your_repo}/Gnu-RL" # Path to Gnu-RL directory

# Note: __file__ now refers to this script (online_learning_main.py). 
# This should work correctly if both scripts are in the same directory.
mpc_path = os.path.abspath(os.path.join(__file__,'..', '..'))

# Add all paths to the system path so Python can find the modules
sys.path.append(repo_path)
sys.path.append(BOPTEST_GYM_PATH)
sys.path.insert(0, mpc_path)

from online_learning import test_loop, imit_dir_gen
if __name__ == "__main__":

    class Args:
        gamma = 0.98  # Discount factor
        state_lr = 5e-3 # Learning rate for state-space parameters
        action_lr = 5e-1  # Learning rate for action cost parameters
        update_episode = 1  # PPO update episode; if -1, do not update weights
        T = 24  # Planning horizon
        step = 1800  # Time step in simulation, unit in seconds (default: 900 for 15 minutes)
        save_name = 'rl'  # Save name
        sigma_init = 0.02 #initial sigma for exploration
        run = 1 #seed for each run. we override this for executing multiple runs in a loop.
        folder_name = 'O_R_CostCalibrated2' #folder name to save the results

        # Cost function parameters. 
        # These are only used if read_R and read_O are False. 
        # Change them if you want to use different initial values OR if you don't want to do cost calibration.
        R_hat = 0.000001  # Hyperparameter for balancing comfort and energy. Assigning a really small value, makes the action cost be linear.
        O_hat = 1  # occupied comfort weight.
        O_unocc = 0.0001 # unoccupied comfort weight.
        s_t = 0.1 # this is the linear cost coefficient for the state cost. We keep it constant since R_hat and O_hat are learned automatically.
        cost_calibration = True # Whether to calibrate the cost parameters or not. Do this if you want to use a non-quadratic reward function.

        ### Imitation learning parameters
        IL_ETA_FOR_PATH = 0.1 # chosen eta (O_occ) value for the imitation learning 
        IL_LR_FOR_PATH = 0.005 # chosen learning rate value for the imitation learning 
        IL_LAMDA_FOR_PATH = 0.1 # chosen lamda value for the imitation learning
        IL_FHAT_EPOCH = 49 # chosen epoch for the imitation learning
        init = 4 # initialization run value for the chosen imitation learning run. [we execute imitation learning for 10 runs, this is the one that gave the best result]
        
        ### ABLATION STUDY PARAMETERS
        # Below should be same for regular IbexRL runs. 
        read_R = True  # Read R from the imitation learning results. If False, use values given above.
        read_O = True  # Read O_occ from the imitation learning results. If False, use values given above.
        no_constraint = True # For ablation study, assigning True, will remove the constraints on parameter values
        predecided_init = False  # For ablation study, assigning True, will provide chosen initial estimates. Assigning False will randomly select from predefined bounds.
        explore = True #whether to add exploration noise or not


        

    args = Args()
    args.gnu_rl_path = gnu_rl_path 
    for run in range(1,11):  # Run 10 test cases with different seeds
        args.run = run
        print(f"Running test case {args.run}...")
        torch.manual_seed(args.run)

        # Generate directory names based on the arguments
        imit_dir = imit_dir_gen(args)

        test_loop(args)