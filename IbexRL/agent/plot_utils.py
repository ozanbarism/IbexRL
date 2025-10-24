import os
import sys
import numpy as np
import torch
import pandas as pd
from datetime import datetime

import os
import sys
from datetime import datetime  # Import the datetime module

import gym
import warnings

# Suppress specific warnings related to PyTorch
warnings.filterwarnings("ignore", message="indexing with dtype torch.uint8 is now deprecated")
warnings.filterwarnings("ignore", message="To copy construct from a tensor, it is recommended to use")


# Assign mpc_path directly
mpc_path = os.path.abspath("/Users/ozanbaris/Documents/GitHub/EnergyWeek/IbexRLAnimation")  # Adjust this path based on your file structure
sys.path.insert(0, mpc_path)

from diff_mpc import mpc
from diff_mpc.mpc import QuadCost, LinDx

repo_path = os.path.abspath("/Users/ozanbaris/Documents/GitHub/EnergyWeek/IbexRLAnimation/agent")  # Adjust this path as necessary to locate your repo's root
sys.path.insert(0, repo_path)

import argparse
import numpy as np
import pandas as pd
import copy
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.distributions import MultivariateNormal, Normal

from agent.utils import make_dict, Replay_Memory, Dataset, cumulative_reward, generate_daily_setpoint_schedule, ensure_3d

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE


def load_dataset():
    # Read historical data
    dataset = pd.read_csv(os.path.join(repo_path, "RLC_data.csv"))
    dataset['Time'] = pd.to_datetime(dataset['Time'])

    # Calculate COP
    theta = dataset['temperature (degC)']
    dataset['COP'] = 0.0008966 * theta**2 + 0.1074 * theta + 3.098

    # Compute P and deltaT
    dataset['P'] = dataset['AHU_main'] + dataset['AHU_Aux'] + dataset['AC_unitout']
    dataset['deltaT'] = dataset['temperature (degC)'] - dataset['temperature_ebtron']

    # Convert surface solar radiation
    dataset['surfacesolarradiation (kW/m^2)'] = dataset['surfacesolarradiation (W/m^2)'] / 1000


    # limit data to from '2023-12-15' to '2023-12-30'
    dataset = dataset[(dataset['Time'] >= '2024-01-01') & (dataset['Time'] < '2024-01-15')]
    step=3600
    # Set Time as the index for easy datetime access
    dataset.set_index('Time', inplace=True)
    # Resample the dataset to a new timestep if `args.step` is not 300 seconds
    if step != 300:
        dataset = dataset.resample(f'{step}S').mean()

    return dataset


class CustomEnv():
    def __init__(self, noise_std=0.5, epoch=48):
        """
        Initialize the environment.
        Args:
            noise_std (float): Standard deviation of the Gaussian noise.
        """
        self.noise_std = noise_std
        # Save dataset for future use
        self.dataset = load_dataset()
        self.n_state = 1
        self.n_ctrl = 3
        first_time = self.dataset.index[0]
        self.start_year = first_time.year
        self.start_mon = first_time.month
        self.start_day = first_time.day

        hyperparameters = "0.05_0.001"
        epoch_model = 48

        #param_dir = f'/Users/ozanbaris/Documents/GitHub/RLC/agent/results_{hyperparameters}'
        #parameters = {
        #    "Rm": np.load(f"{param_dir}/parameters/Rm-{epoch}.npy"),
        #    "Rout": np.load(f"{param_dir}/parameters/Rout-{epoch}.npy"),
        #    "Capacitance": np.load(f"{param_dir}/parameters/C-{epoch}.npy"),
        #    "Tm": np.load(f"{param_dir}/parameters/Tm-{epoch}.npy"),
        #    "Ai": np.load(f"{param_dir}/parameters/Ai-{epoch}.npy"),
        #    "eta_aux": np.load(f"{param_dir}/parameters/eta_aux-{epoch}.npy"),
        #    "O_hat": 0.1,
        #    "R_hat": np.array([0.1, 5, 5])
        #}
        #print("Parameters Ohat and Rhat", parameters["O_hat"], parameters["R_hat"])

        #self.Rm = parameters["Rm"]
        #self.Rout = parameters["Rout"]
        #self.C = parameters["Capacitance"]
        #self.Tm = parameters["Tm"]
        #self.Ai = parameters["Ai"]
        #self.eta_aux = parameters["eta_aux"]


        # Read parameters
        self.Rm = 1.06
        self.Rout = 2.04
        self.C = 7.5 * 3600
        self.Tm = 20.6
        self.Ai = 0.01
        self.eta_aux = 0.8
        #make them tensors
        self.Rm = torch.tensor(self.Rm, dtype=torch.double)
        self.Rout = torch.tensor(self.Rout, dtype=torch.double)
        self.C = torch.tensor(self.C, dtype=torch.double)
        self.Tm = torch.tensor(self.Tm, dtype=torch.double)
        self.Ai = torch.tensor(self.Ai, dtype=torch.double)
        self.eta_aux = torch.tensor(self.eta_aux, dtype=torch.double)
        
        # Define arguments directly
        class Args:
            gamma = 0.98  # Discount factor
            seed = 42  # Random seed
            action_lr = 5e-2  # Learning rate
            state_lr = 5e-1 # Learning rate
            update_episode = 1  # PPO update episode; if -1, do not update weights
            T = 24  # Planning horizon
            step = 3600  # Time step in simulation, unit in seconds (default: 900 for 15 minutes)
            save_name = 'rl'  # Save name
            eta = 1  # Hyperparameter for balancing comfort and energy
            load_imitation = True
        
        args=Args()

        self.update_matrices()



    def update_matrices(self):
        # Define arguments directly
        class Args:
            gamma = 0.98  # Discount factor
            seed = 42  # Random seed
            action_lr = 5e-2  # Learning rate
            state_lr = 5e-1 # Learning rate
            update_episode = 1  # PPO update episode; if -1, do not update weights
            T = 24  # Planning horizon
            step = 3600  # Time step in simulation, unit in seconds (default: 900 for 15 minutes)
            save_name = 'rl'  # Save name
            eta = 1  # Hyperparameter for balancing comfort and energy
            load_imitation = True
        args=Args()

        #read Rm value from a txt file called Rm.txt
        with open("Rm.txt", "r") as f:
            Rm = f.read()
            #convert to float 
            Rm = float(Rm)
        #print("Rm is read from the file as", Rm)
        #assign that value to self.Rm
        self.Rm = torch.tensor(Rm, dtype=torch.double)

        # Calculate `Ac` based on Rm and Rout, ensuring it’s connected to autograd
        # Clone parameters to avoid in-place issues
        C_cloned = self.C.clone()
        Rm_cloned = self.Rm.clone()
        Rout_cloned = self.Rout.clone()

        # Perform the computation using cloned tensors
        Ac_new = (-1 / (C_cloned * Rm_cloned) - 1 / (C_cloned * Rout_cloned))

        # Clone the result to ensure it is detached from any computational aliasing
        self.Ac = Ac_new.clone()

        self.Ac = self.Ac.unsqueeze(0).unsqueeze(1)

        # Define `Bu` based on the number of control actions, using autograd-friendly tensors
        if self.n_ctrl == 1:
            self.Bu = torch.tensor([[1 / self.C]], dtype=torch.double)
        elif self.n_ctrl == 3:
            # Clone tensors to avoid in-place issues
            C_cloned = self.C.clone()
            eta_aux_cloned = self.eta_aux.clone()

            # Perform computations using the cloned tensors
            Bu_new = torch.stack([1 / C_cloned, eta_aux_cloned / C_cloned, eta_aux_cloned / C_cloned], dim=0).unsqueeze(0)

            # Assign the result to self.Bu
            self.Bu = Bu_new



            # Clone the parameters to avoid in-place issues
            C_cloned = self.C.clone()
            Ai_cloned = self.Ai.clone()

            # Perform the computation using the cloned tensors
            Bd_new = torch.stack([1 / C_cloned, 1 / C_cloned, Ai_cloned / C_cloned], dim=0).unsqueeze(0)

            # Assign the computed tensor to self.Bd
            self.Bd = Bd_new.clone()


        # Calculate `F` using matrix exponential of `Ac`
        self.F = torch.matrix_exp(self.Ac * args.step)

        # Calculate `G_u` and `Bd_hat` with error handling for inversion
        try:
            Ac_inv = torch.inverse(self.Ac)
            self.G_u = torch.matmul(Ac_inv, torch.matmul(self.F - torch.eye(self.n_state, dtype=torch.double), self.Bu))
            self.Bd_hat = torch.matmul(Ac_inv, torch.matmul(self.F - torch.eye(self.n_state, dtype=torch.double), self.Bd))
        except RuntimeError:
            # Use pseudoinverse for singular matrices
            Ac_inv = torch.pinverse(self.Ac)
            self.G_u = torch.matmul(Ac_inv, torch.matmul(self.F - torch.eye(self.n_state, dtype=torch.double), self.Bu))
            self.Bd_hat = torch.matmul(Ac_inv, torch.matmul(self.F - torch.eye(self.n_state, dtype=torch.double), self.Bd))

        # Combine `F` and `G_u` to form `F_hat`
        self.F_hat = torch.cat([self.F, self.G_u], dim=1)

    def reset(self):
        """
        Reset the environment to an initial state.
        Returns:
            tuple: (cur_time, obs, is_terminal)
        """
        obs = self.dataset[['temperature_ebtron', 'temperature (degC)',
                            'surfacesolarradiation (kW/m^2)', 'COP']].values[0, :]
        cur_time = self.dataset.index[0]
        is_terminal = False

        return cur_time, obs, is_terminal

    def step(self, temp, action, cur_time):
        """
        Take a step in the environment.
        Args:
            temp (list): Current temperature state.
            action (torch.Tensor): Action vector.
            cur_time (datetime): Current time.
        Returns:
            tuple: (next_state, reward, done, info)
        """
        # Extract state and disturbance variables
        #print("action in the environment", action)
        dist_name = ['temperature (degC)', 'surfacesolarradiation (kW/m^2)']
        state = torch.tensor(temp).unsqueeze(1).double()
        dist = torch.tensor(self.dataset.loc[cur_time, dist_name]).unsqueeze(1).double()
        cop = torch.tensor(self.dataset.loc[cur_time, 'COP'])

        #update
        self.update_matrices()

        # Prepare disturbances for the prediction step
        Tm_tensor = torch.tensor(self.Tm).unsqueeze(0).unsqueeze(1)
        dt = torch.cat((Tm_tensor, dist), dim=0)

 
        # Calculate `ft` based on `Bd_hat` and disturbances
        ft = torch.mm(self.Bd_hat, dt)
        #print("env ft", ft)
        # Create an adjusted version of F_hat for this prediction step
        adjusted_F_hat = self.F_hat.clone().detach()
        adjusted_F_hat[0, 1] *= cop  # Modify F_hat with COP value
        #print("adjusted_F_hat", adjusted_F_hat)
        # Concatenate initial state and action for next state prediction
        tau = torch.cat([state, action.double().T], dim=0)

        # Calculate the next state using adjusted_F_hat
        next_state = torch.mm(adjusted_F_hat, tau) + ft

        # Add Gaussian noise to the next state

        # Determine reward, termination, and any additional info
        done = False  # Example: No terminal condition
        info = {"time": cur_time}

        timeStep = 3600

        # Update the current time
        cur_time = cur_time + pd.Timedelta(seconds=timeStep)

        # Extract the data for the new cur_time from self.dataset for columns 'temperature (degC)', 'surfacesolarradiation (kW/m^2)', 'COP'
        next_oat = self.dataset.loc[cur_time, 'temperature (degC)']
        next_solar_radiation = self.dataset.loc[cur_time, 'surfacesolarradiation (kW/m^2)']
        next_COP = self.dataset.loc[cur_time, 'COP']

        # Flatten next_state to 1D or extract specific scalar values if required
        next_state_values = next_state.detach().numpy().flatten()

        # Ensure all elements are scalars or 1D arrays
        obs = np.array([*next_state_values, next_oat, next_solar_radiation, next_COP])

        return timeStep, obs, done





class IbexRL():
    def __init__(self, memory, T, n_ctrl, n_state, target, disturbance, u_upper, u_lower, parameters, clip_param = 0.1):
        self.memory = memory
        self.clip_param = clip_param
        class Args:
            gamma = 0.98  # Discount factor
            seed = 42  # Random seed
            action_lr = 5e-2  # Learning rate
            state_lr = 1e-1 # Learning rate
            update_episode = 1  # PPO update episode; if -1, do not update weights
            T = 24  # Planning horizon
            step = 3600  # Time step in simulation, unit in seconds (default: 900 for 15 minutes)
            save_name = 'rl'  # Save name
            eta = 1  # Hyperparameter for balancing comfort and energy
            load_imitation = True
            
        args=Args()
        self.T = T
        self.step = 3600
        self.n_ctrl = n_ctrl
        self.n_state = n_state
        
        self.target = target
        self.dist = disturbance
        self.n_dist = self.dist.shape[1]
        #print("parameters", parameters)
        #read parameters
        Rm = parameters["Rm"]
        Rout = parameters["Rout"]
        Capacitance = parameters["Capacitance"]
        Tm = parameters["Tm"]
        Ai = parameters["Ai"]
        eta_aux = parameters["eta_aux"]
        O_hat = parameters["O_hat"]
        R_hat = parameters["R_hat"]

        #assign requires grad to all parameters
        self.Rm = torch.tensor(Rm).double().requires_grad_()
        self.Rout = torch.tensor(Rout).double().requires_grad_()
        self.C = torch.tensor(Capacitance).double().requires_grad_()
        self.Tm = torch.tensor(Tm).double().requires_grad_()
        self.Ai = torch.tensor(Ai).double().requires_grad_()
        self.eta_aux = torch.tensor(eta_aux).double().requires_grad_()
        self.O_hat = torch.tensor(O_hat).double().requires_grad_()
        self.R_hat = torch.tensor(R_hat).double().requires_grad_()

        #print("Ohat", self.O_hat)
        #print("R_hat", self.R_hat)
        self.update_matrices()

        self.state_optimizer = optim.Adam([self.Rm, self.Rout, self.C, self.Tm, self.Ai, self.eta_aux], lr=args.state_lr)
        self.action_optimizer = optim.Adam([self.O_hat, self.R_hat], lr=args.action_lr)

        if n_ctrl == 1:
            self.u_lower = torch.full((T, 1, n_ctrl), u_lower, dtype=torch.double)
            self.u_upper = torch.full((T, 1, n_ctrl), u_upper, dtype=torch.double)
        else:
            # Convert u_lower and u_upper to tensors and expand to match the desired shape
            self.u_lower = torch.tensor(u_lower, dtype=torch.double).expand(T, 1, n_ctrl)
            self.u_upper = torch.tensor(u_upper, dtype=torch.double).expand(T, 1, n_ctrl)

    def update_matrices(self):
        class Args:
            gamma = 0.98  # Discount factor
            seed = 42  # Random seed
            action_lr = 5e-2  # Learning rate
            state_lr = 1e-1 # Learning rate
            update_episode = 1  # PPO update episode; if -1, do not update weights
            T = 24  # Planning horizon
            step = 3600  # Time step in simulation, unit in seconds (default: 900 for 15 minutes)
            save_name = 'rl'  # Save name
            eta = 1  # Hyperparameter for balancing comfort and energy
            load_imitation = True
            

        hyperparameters = "0.05_0.001"
        epoch_model = 48
        args = Args()
        # Calculate `Ac` based on Rm and Rout, ensuring it’s connected to autograd
        # Clone parameters to avoid in-place issues
        C_cloned = self.C.clone() #* 3600
        Rm_cloned = self.Rm.clone()
        Rout_cloned = self.Rout.clone()

        # Perform the computation using cloned tensors
        Ac_new = (-1 / (C_cloned * Rm_cloned) - 1 / (C_cloned * Rout_cloned))

        # Clone the result to ensure it is detached from any computational aliasing
        self.Ac = Ac_new.clone()

        self.Ac = self.Ac.unsqueeze(0).unsqueeze(1)

        # Define `Bu` based on the number of control actions, using autograd-friendly tensors
        if self.n_ctrl == 1:
            self.Bu = torch.tensor([[1 / self.C]], dtype=torch.double)
        elif self.n_ctrl == 3:
            # Clone tensors to avoid in-place issues

            eta_aux_cloned = self.eta_aux.clone()
            # Perform computations using the cloned tensors
            Bu_new = torch.stack([1 / C_cloned, eta_aux_cloned / C_cloned, eta_aux_cloned / C_cloned], dim=0).unsqueeze(0)

            # Assign the result to self.Bu
            self.Bu = Bu_new
            Ai_cloned = self.Ai.clone()

            # Perform the computation using the cloned tensors
            Bd_new = torch.stack([1 / C_cloned, 1 / C_cloned, Ai_cloned / C_cloned], dim=0).unsqueeze(0)

            # Assign the computed tensor to self.Bd
            self.Bd = Bd_new.clone()


        # Calculate `F` using matrix exponential of `Ac`
        self.F = torch.matrix_exp(self.Ac * args.step)

        # Calculate `G_u` and `Bd_hat` with error handling for inversion
        try:
            Ac_inv = torch.inverse(self.Ac)
            self.G_u = torch.matmul(Ac_inv, torch.matmul(self.F - torch.eye(self.n_state, dtype=torch.double), self.Bu))
            self.Bd_hat = torch.matmul(Ac_inv, torch.matmul(self.F - torch.eye(self.n_state, dtype=torch.double), self.Bd))
        except RuntimeError:
            # Use pseudoinverse for singular matrices
            Ac_inv = torch.pinverse(self.Ac)
            self.G_u = torch.matmul(Ac_inv, torch.matmul(self.F - torch.eye(self.n_state, dtype=torch.double), self.Bu))
            self.Bd_hat = torch.matmul(Ac_inv, torch.matmul(self.F - torch.eye(self.n_state, dtype=torch.double), self.Bd))

        # Combine `F` and `G_u` to form `F_hat`
        self.F_hat = torch.cat([self.F, self.G_u], dim=1)


    # Use the "current" flag to indicate which set of parameters to use
    def forward(self, x_init, F_hat_repeated, ft, C, c, n_batch, current = True, n_iters=20):
        T, n_batch, n_dist = ft.shape
        #print(f"Before update_matrices: self.C version: {self.C._version}, self.eta_aux version: {self.eta_aux._version}")
        self.update_matrices()
        #print(f"After update_matrices: self.C version: {self.C._version}, self.eta_aux version: {self.eta_aux._version}")

        # Correct repetition for u_lower and u_upper
        u_lower = self.u_lower.repeat(1, n_batch, 1)  # [T, n_batch, n_ctrl]
        u_upper = self.u_upper.repeat(1, n_batch, 1)  # [T, n_batch, n_ctrl]

        x_lqr, u_lqr, objs_lqr = mpc.MPC(n_state=self.n_state,
                                         n_ctrl=self.n_ctrl,
                                         T=self.T,
                                         u_lower= u_lower,
                                         u_upper= u_upper,
                                         lqr_iter=n_iters,
                                         backprop = True,
                                         verbose=0,
                                         exit_unconverged=False,
                                         )(x_init.double(), QuadCost(C.double(), c.double()),
                                           LinDx(F_hat_repeated, ft.double()))
        return x_lqr, u_lqr


    def Dist_func(self, d, current = False):
        
        if current: # d in n_batch x n_dist x T-1
            # Calculate `ft` based on `Bd_hat` and disturbances
            self.update_matrices()
            n_batch = d.shape[0]
            #print("n_batch", n_batch)

            # Reshape Tm_tensor to align as an additional column
            Tm_tensor = self.Tm.expand(n_batch, self.T - 1).unsqueeze(1)  # [24, 1, 23]
            #print("Tm_tensor size", Tm_tensor.size())
            #print("d size", d.size())

            # Concatenate Tm_tensor with d along the second dimension (axis=1)
            dt = torch.cat((Tm_tensor, d), dim=1)  # [24, 3, 23]
            #print("dt size", dt.size())

            # Perform the batch matrix multiplication
            ft = torch.bmm(self.Bd_hat.repeat(n_batch, 1, 1), dt)  # n_batch x n_state x T-1


            ft = ft.transpose(1,2) # n_batch x T-1 x n_state
            ft = ft.transpose(0,1) # T-1 x n_batch x n_state
        else: # d in n_dist x T-1
            #print("shape of d in dist_func", d.shape)
            # Repeat `Tm` and concatenate with disturbances
            Tm_tensor = self.Tm.expand(1, self.T - 1)
            #print("Tm_tensor", Tm_tensor)
            dt = torch.cat((Tm_tensor, d), dim=0)

            ft = torch.mm(self.Bd_hat, dt).transpose(0, 1) # T-1 x n_state
            ft = ft.unsqueeze(1) # T-1 x 1 x n_state
        return ft

    def Infuse_COP(self, d, current=False):
        if current:
            self.update_matrices()
            F_hat_to_use = self.F_hat
            n_batch = d.shape[0]  # Extract the batch size (3)

            # Extract outdoor air temperature values
            theta = d[:, 0, :]  # Shape: [n_batch, T]
            cop_values_raw = 0.0008966 * theta**2 + 0.1074 * theta + 3.098  # Shape: [n_batch, T]

            T = self.T

            # Repeat F_hat to match T-1 steps and n_batch
            F_hat_repeated = F_hat_to_use.repeat(1, n_batch, 1, 1)  # Shape: [1, n_batch, dim1, dim2]
            F_hat_repeated = F_hat_repeated.repeat(T - 1, 1, 1, 1)  # Shape: [T-1, n_batch, dim1, dim2]

            # Extract the next T-1 COP values for the prediction horizon
            cop_values_raw = cop_values_raw[:, : T - 1]  # Shape: [n_batch, T-1]
            cop_values = cop_values_raw.transpose(1, 0).unsqueeze(-1)  # Shape: [T-1, n_batch, 1]

            # Multiply the second element in the last dimension of F_hat_repeated by the corresponding COP values
            F_hat_repeated[:, :, 0, 1] *= cop_values.squeeze(-1)  # Targeting the heating input for each batch

        else:
            theta = d[0, :]  # Extract the first element (shape: [T])
            cop_values_raw = 0.0008966 * theta**2 + 0.1074 * theta + 3.098  # Shape: [T]

            T = self.T

            F_hat_to_use = self.F_hat

            F_hat_repeated = F_hat_to_use.repeat(T - 1, 1, 1, 1)  # Shape: [T-1, 1, dim1, dim2]

            # Extract the next T-1 COP values for the prediction horizon
            cop_values_raw = cop_values_raw[: T - 1]  # Shape: [T-1]
            cop_values = cop_values_raw.view(T - 1, 1, 1)  # Shape: [T-1, 1, 1]

            # Multiply the second element in the last dimension of F_hat_repeated by the corresponding COP values
            F_hat_repeated[:, :, 0, 1] *= cop_values.squeeze(-1)  # Targeting the heating input

        return F_hat_repeated

    def select_action(self, mu, sigma):
        if self.n_ctrl > 1:
            sigma_sq = torch.ones(mu.size()).double() * sigma**2
            dist = MultivariateNormal(mu, torch.diag(sigma_sq.squeeze()).unsqueeze(0))
        else:
            dist = Normal(mu, torch.ones_like(mu) * sigma)
        
        action = dist.sample()
        action = torch.clamp(action, min=0)  # Clamp to ensure actions are non-negative
        
        log_prob = dist.log_prob(action)
        return action, log_prob

    
    def update_state_parameters(self, loader):
        for i in range(1):  # Single iteration
            for states, actions, next_states, dist, old_reward, targets, C, c in loader:
                n_batch = states.shape[0]

                # Compute the disturbance function
                f = self.Dist_func(dist, current=True)  # T-1 x n_batch x n_state
                F_hat_repeated = self.Infuse_COP(dist, current=True)

                # Compute cost function
                C, c = self.Cost_function(targets=targets, n_batch=n_batch)

                # Debug cost function outputs
                print("Cost function outputs:")
                print("self.R_hat:", self.R_hat)
                print("self.O_hat:", self.O_hat)
                # Forward pass for optimization
                opt_states, opt_actions = self.forward(states, F_hat_repeated, f, C, c, n_batch, current=True)
                #opt_states, opt_actions = self.forward(states, F_hat_repeated, f, C.transpose(0, 1), c.transpose(0, 1), n_batch, current = True) # x, u: T x N x Dim.
                # Create tau by concatenating states and actions
                tau = torch.cat([states, actions], dim=-1)  # Shape: [24, 4]

                fhat_reduced = F_hat_repeated[0]  # Shape: [4, 1, 4] n_batch x (n_state + n_ctrl)
                f_reduced = f[0]  # Shape: [4, 1]

                # Step 3: Batched matrix multiplication
                nState_est = torch.bmm(fhat_reduced, tau.unsqueeze(-1)).squeeze(-1) + f_reduced  # Shape: [24, 1]

                #print("fhat_repeated", F_hat_repeated)
                self.update_matrices()


                # Clone outputs to protect the computation graph
                opt_states = opt_states.clone()
                opt_actions = opt_actions.clone()

                # Enable anomaly detection
                torch.autograd.set_detect_anomaly(True)

                # Define a threshold for MSE loss below which no updates are made
                mse_loss_threshold = 1e-6  # Adjust this value as needed

                # Compute MSE loss for state updates
                mse_loss = torch.mean((nState_est - next_states) ** 2)
                self.mse_loss = mse_loss
                # Debug MSE loss
                print(f"MSE loss value: {mse_loss.item()}, grad_fn: {mse_loss.grad_fn}")

                # Check if the loss exceeds the threshold
                if mse_loss.item() > mse_loss_threshold:
                    # State optimization
                    self.state_optimizer.zero_grad()
                    mse_loss.backward(retain_graph=True)
                    self.state_optimizer.step()
                else:
                    print("Skipping parameter update due to small loss.")

            self.update_matrices()
    def update_action_parameters(self, state, targets, dist, max_iterations=100, tol=1e-4):
        n_batch = state.shape[0]

        # Optimization loop
        for iteration in range(max_iterations):
            #print(f"Iteration {iteration + 1}/{max_iterations}")

            # Recompute the disturbance function (ensures a fresh graph for each iteration)
            f = self.Dist_func(dist)  # T-1 x n_batch x n_state
            F_hat_repeated = self.Infuse_COP(dist)

            # Recompute the cost function for the current state
            C, c = self.Cost_function(targets=targets, n_batch=n_batch)

            # Forward pass for optimization
            opt_states, opt_actions = self.forward(state, F_hat_repeated, f, C, c, n_batch, current=True)

            # Compute the reward based on the current parameters
            reward = cumulative_reward(opt_states, targets, opt_actions)
            #print(f"Reward: {reward.item()}")

            # Define the loss (negative reward to maximize reward)
            loss = -reward / 1000
            #print(f"Loss: {loss.item()}")
            
            # Backpropagation
            self.action_optimizer.zero_grad()  # Clear previous gradients
            loss.backward()  # Compute gradients for the current loss
            self.action_optimizer.step()  # Update parameters

            # Clamp R_hat and O_hat to avoid instability
            with torch.no_grad():
                self.R_hat.clamp_(min=1e-9)
                self.O_hat.clamp_(min=1e-9)

            # Update matrices based on the new parameters
            self.update_matrices()

            # Check for convergence
            if abs(loss.item()) < tol:
                print(f"Converged after {iteration + 1} iterations with loss: {loss.item()}")
                break

        print(f"Final Parameters: O_hat: {self.O_hat}, R_hat: {self.R_hat}")


    def Cost_function(self, targets=None, n_batch=None):
        # Add an extra dimension to O_hat to make it compatible for concatenation
        C_diag = torch.cat([self.O_hat.unsqueeze(0), self.R_hat])

        T = self.T
        # Create a diagonal matrix from the concatenated vector
        C_hat = torch.diag(C_diag)
        
        # Store for debugging
        self.current_C_hat = C_hat
        
        # Clone C_hat to ensure no aliasing and safe propagation of gradients
        C_hat_cloned = C_hat.clone() 

        # Perform unsqueeze and repeat without overwriting the tensor directly
        C_cost_new = C_hat_cloned.unsqueeze(0).repeat(T, 1, 1).unsqueeze(1)

        # Assign to self.C_cost to avoid in-place modification
        self.C_cost = C_cost_new.clone()

        # If targets are not None, adjust `C_cost` based on target values
        if targets is not None:
            x_target = ensure_3d(targets, n_batch)  # Ensure targets has 3 dimensions
            # Reshape x_target for broadcasting: [n_batch, 1, T] -> [T, n_batch, 1]
            x_target_reshaped = x_target.permute(2, 0, 1)  # [T, n_batch, 1]

            # Identify indexes where target values are 18
            target_indexes = (x_target_reshaped.squeeze(-1) == 18)  # [T, n_batch]

            # Initialize a reduction factor (e.g., 0.5) for `O` values corresponding to target 18
            reduction_factor = 1

            # Update C_cost for the batch dimension: [T, 1, 4, 4] -> [T, n_batch, 4, 4]
            self.C_cost = self.C_cost.repeat(1, n_batch, 1, 1)  # Repeat along batch dimension

            # Adjust O_hat for these target indexes in C_cost
            for t in range(T):
                for b in range(n_batch):
                    if target_indexes[t, b]:
                        # Scale the corresponding O value in the diagonal of `C_cost`
                        self.C_cost[t, b, :self.n_state, :self.n_state] *= reduction_factor

            # Initialize c with zeros: [T, n_batch, 4]
            c = torch.zeros(T, n_batch, self.n_state + self.n_ctrl, dtype=torch.double)

            # Compute c using broadcasting
            # Apply reduced O_hat for target indexes corresponding to 18
            for t in range(T):
                for b in range(n_batch):
                    if target_indexes[t, b]:
                        c[t, b, :self.n_state] = -self.O_hat * reduction_factor * x_target_reshaped[t, b]
                    else:
                        c[t, b, :self.n_state] = -self.O_hat * x_target_reshaped[t, b]

        return self.C_cost, c


