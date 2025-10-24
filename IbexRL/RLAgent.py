import os
import sys
from datetime import datetime  # Import the datetime module
import csv
import gym
import warnings

# Suppress specific warnings related to PyTorch
warnings.filterwarnings("ignore", message="indexing with dtype torch.uint8 is now deprecated")
warnings.filterwarnings("ignore", message="To copy construct from a tensor, it is recommended to use")


# Assign mpc_path directly
mpc_path = os.path.abspath("/Users/ozanbaris/Documents/GitHub/Ibex_RL/Ibex_RL")  # Adjust this path based on your file structure
sys.path.insert(0, mpc_path)

from diff_mpc import mpc
from diff_mpc.mpc import QuadCost, LinDx

repo_path = os.path.abspath("/Users/ozanbaris/Documents/GitHub/Ibex_RL/Ibex_RL")  # Adjust this path as necessary to locate your repo's root
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

from agent.utils import Replay_Memory, Dataset, cumulative_reward, generate_daily_setpoint_schedule, ensure_3d

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE

# Define arguments directly
class Args:
    gamma = 0.98  # Discount factor
    seed = 42  # Random seed
    action_lr = 5e-2  # Learning rate
    state_lr = 5e-3  # Learning rate
    update_episode = 1  # PPO update episode; if -1, do not update weights
    T = 24  # Planning horizon
    step = 3600  # Time step in simulation, unit in seconds (default: 900 for 15 minutes)
    save_name = 'rl'  # Save name
    eta = 1  # Hyperparameter for balancing comfort and energy

args = Args()

class IbexRL():
    def __init__(self, memory, T, n_ctrl, n_state,  u_upper, u_lower, parameters, clip_param = 0.1):
        self.memory = memory
        self.clip_param = clip_param
        
        self.T = T
        self.step = args.step
        self.n_ctrl = n_ctrl
        self.n_state = n_state
        

        self.n_dist = 2

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



    
    def update_parameters(self, loader):
        for i in range(1):  # Single iteration
            for states, actions, next_states, dist, old_reward, targets in loader:
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

                # Create tau by concatenating states and actions
                tau = torch.cat([states, actions], dim=-1)  # Shape: [24, 4]

                fhat_reduced = F_hat_repeated[0]  # Shape: [4, 1, 4] n_batch x (n_state + n_ctrl)
                f_reduced = f[0]  # Shape: [4, 1]

                # Step 3: Batched matrix multiplication
                nState_est = torch.bmm(fhat_reduced, tau.unsqueeze(-1)).squeeze(-1) + f_reduced  # Shape: [24, 1]

    
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
                # Save MSE loss with the current time (without seconds) to a CSV file
                # Get the current time (without seconds)
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M")

                # Save MSE loss to CSV
                with open('mse_loss_history.csv', mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([current_time, mse_loss.item()])

                # Debug MSE loss
                print(f"MSE loss value: {mse_loss.item()}, grad_fn: {mse_loss.grad_fn}")

                # Check if the loss exceeds the threshold
                if mse_loss.item() > mse_loss_threshold:
                    # State optimization
                    self.state_optimizer.zero_grad()
                    mse_loss.backward(retain_graph=True)
                    self.state_optimizer.step()
                    print("Parameters updated.")
                else:
                    print("Skipping parameter update due to small loss.")

                # Action optimization
                self.action_optimizer.zero_grad()
                # Infer batch size from the size of targets
                batch_size = targets.size(0)  # Get the first dimension as the batch size

                # Initialize a list to store rewards for each batch element
                batch_rewards = []

                # Iterate through each batch element
                for i in range(batch_size):
                    # Extract the 24x1 slices for the current batch
                    opt_state_slice = opt_states[:, i, :]  # Shape: [24, 1]
                    opt_actions_slice = opt_actions[:, i, :]  # Shape: [24, 1]
                    target_slice = targets[i, :, :]       # Shape: [1, 24]. Transpose to match if needed
                    
                    # Calculate the reward for this specific batch element
                    reward = cumulative_reward(opt_state_slice, target_slice, opt_actions_slice)
                    batch_rewards.append(reward)

                # Combine rewards (e.g., sum or average)
                total_reward = torch.stack(batch_rewards).sum()  # or .mean() if you want average

                # Define the loss
                loss = -total_reward / 1000
                loss.backward()
                self.action_optimizer.step()

                # Safely clamp R_hat without affecting autograd
                with torch.no_grad():
                    self.R_hat.clamp_(min=1e-9)
                    self.O_hat.clamp_(min=1e-9)


            #self.F_hat_old = self.F_hat.detach().clone()
            #self.Bd_hat_old = self.Bd_hat.detach().clone()
            #self.O_hat_old = self.O_hat.detach().clone()
            #self.R_hat_old = self.R_hat.detach().clone()
            
            self.update_matrices()
            #print(F"UPDATED PARAMETERS: O_hat {self.O_hat}, R_hat {self.R_hat}")

    def Cost_function_old(self, targets=None, n_batch=None):
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

        #self.C_cost = C_hat.unsqueeze(0).repeat(T, 1, 1).unsqueeze(1)  # [T, 1, 4, 4]

        # If targets are not None, calculate `c` using the new shape of `x_target`
        if targets is not None:
            x_target = ensure_3d(targets, n_batch)  # Ensure targets has 3 dimensions
            # Reshape x_target for broadcasting: [n_batch, 1, T] -> [T, n_batch, 1]
            x_target_reshaped = x_target.permute(2, 0, 1)  # [T, n_batch, 1]

            # Initialize c with zeros: [T, n_batch, 4]
            c = torch.zeros(T, n_batch, self.n_state + self.n_ctrl, dtype=torch.double)

            # Compute c using broadcasting
            c[:, :, :self.n_state] = -self.O_hat * x_target_reshaped  # Broadcasted multiplication

            # Update C_cost for the batch dimension: [T, 1, 4, 4] -> [T, n_batch, 4, 4]
            self.C_cost = self.C_cost.repeat(1, n_batch, 1, 1)  # Repeat along batch dimension

        return self.C_cost, c
    
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
