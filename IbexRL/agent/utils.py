# Helper Functions
import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
import math
from datetime import datetime, timedelta
def make_dict(obs_name, obs):
    zipbObj = zip(obs_name, obs)
    return dict(zipbObj)

def R_func(obs_dict, action, eta_occ, s_t):
    reaTSetHea_y = round(obs_dict['reaTSetHea_y'])
    occupied = (reaTSetHea_y == 21)
    #print("reaTSetHea_y value:", obs_dict['reaTSetHea_y'], "Type:", type(obs_dict['reaTSetHea_y']))
    #print("reaTSetHea_y rounded:", reaTSetHea_y, "Occupied:", occupied, "coeff", eta_occ[int(occupied)])
    reward = - 0.5 * eta_occ[int(occupied)] * (obs_dict['reaTZon_y'] - obs_dict['reaTSetHea_y'])**2 - s_t*action
    return reward.item()

def non_quadratic_cumulative_reward(states, setpoints, actions, 
                     occupied_penalty, 
                     unoccupied_penalty, 
                     pi_d=0.8, 
                     pi_e=0.15):
    """
    Calculates a cumulative reward with three components:
    1. A linear penalty for temperature deviation, weighted by occupancy.
    2. A penalty for total energy consumption.
    3. A penalty for peak power demand.
    """
    # 1. Determine if each timestep is "occupied" (setpoint is 21)
    is_occupied = (torch.round(setpoints) == 21)

    # 2. Create a tensor of weights based on occupancy
    temp_weights = torch.where(is_occupied, occupied_penalty, unoccupied_penalty)

    # 3. Calculate the weighted, linear temperature deviation penalty
    abs_temp_deviation = torch.abs(states - setpoints)
    temp_deviation_penalty = torch.sum(temp_weights * abs_temp_deviation)

    # 4. Calculate total and max power from the actions
    total_power = torch.sum(actions)
    max_power = torch.max(actions)

    # 5. Compute the final reward using your specified formula
    reward = -pi_e * total_power - pi_d * max_power * 24 - temp_deviation_penalty

    return reward

def ensure_3d(targets, n_batch):
    """
    Ensures that the targets tensor has 3 dimensions.
    If it has 2 dimensions, adds a new batch dimension at the start.
    
    Args:
        targets (torch.Tensor): Input tensor with shape [batch_size, T] or [n_batch, 1, T].
        n_batch (int): The size of the batch to be added if necessary.

    Returns:
        torch.Tensor: A tensor with 3 dimensions [n_batch, 1, T].
    """
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets, dtype=torch.float32)
    if targets.dim() == 2:  # If the tensor has 2 dimensions [batch_size, T]
        targets = targets.unsqueeze(1)  # Add an extra dimension at index 1 -> [batch_size, 1, T]
    elif targets.dim() == 3:
        # No change needed if it already has 3 dimensions
        pass
    else:
        raise ValueError(f"Expected targets to have 2 or 3 dimensions, but got {targets.dim()} dimensions")
    
    if targets.size(0) != n_batch:
        raise ValueError(f"Batch size mismatch: expected {n_batch}, got {targets.size(0)}")

    return targets

def cumulative_reward(opt_states, setpoints, actions):
    # Ensure opt_states, setpoints, and actions have requires_grad=True if necessary

    # Compute power-related values
    power_values = torch.sum(actions, dim=1)  # Total power at each timestep
    max_power = torch.max(power_values)       # Maximum power over all timesteps
    total_power = torch.sum(power_values)     # Total power over all timesteps

    # Compute temperature deviation penalties
    # Separate penalties for warmer and colder deviations
    warmer_deviation = torch.relu(opt_states - setpoints)  # Deviation when opt_state > setpoint
    colder_deviation = torch.relu(setpoints - opt_states)  # Deviation when opt_state < setpoint

    # Penalty factors
    pi_t_warmer = 0.2#0.2
    pi_t_colder = 0.2 #0.2

    # Compute weighted deviation penalties
    temp_deviation_penalty = (
        pi_t_warmer * torch.sum(warmer_deviation) +
        pi_t_colder * torch.sum(colder_deviation)
    )

    # Clamp power values to avoid numerical instability
    max_power = torch.clamp(max_power, min=1e-6)
    total_power = torch.clamp(total_power, min=1e-6)

    # Ensure all intermediate results have requires_grad=True
    assert temp_deviation_penalty.requires_grad, "temp_deviation_penalty lost requires_grad"
    assert max_power.requires_grad, "max_power lost requires_grad"
    assert total_power.requires_grad, "total_power lost requires_grad"

    # Define reward weights
    pi_d = 0.8
    pi_e = 0.15

    # Compute reward
    reward = -pi_e * total_power - pi_d * max_power * 24 - temp_deviation_penalty
    # Check final reward
    assert reward.requires_grad, "Reward lost requires_grad"
    return reward






 
def ensure_3d(targets, n_batch):
    """
    Ensures that the targets tensor has 3 dimensions.
    If it has 2 dimensions, adds a new batch dimension at the start.
    
    Args:
        targets (torch.Tensor): Input tensor with shape [batch_size, T] or [n_batch, 1, T].
        n_batch (int): The size of the batch to be added if necessary.

    Returns:
        torch.Tensor: A tensor with 3 dimensions [n_batch, 1, T].
    """
    if targets.dim() == 2:  # If the tensor has 2 dimensions [batch_size, T]
        targets = targets.unsqueeze(1)  # Add an extra dimension at index 1 -> [batch_size, 1, T]
    elif targets.dim() == 3:
        # No change needed if it already has 3 dimensions
        pass
    else:
        raise ValueError(f"Expected targets to have 2 or 3 dimensions, but got {targets.dim()} dimensions")
    
    if targets.size(0) != n_batch:
        raise ValueError(f"Batch size mismatch: expected {n_batch}, got {targets.size(0)}")

    return targets


def calculate_ppd_pmv(T, day):
    """
    Calculate PPD (Percentage of People Dissatisfied) and PMV (Predicted Mean Vote).
    
    Parameters:
        T (float): Air temperature in °C.
        day (int): 1 for daytime, 0 for nighttime.
    
    Returns:
        tuple: PPD (float), PMV (float)
    """
    # Constants
    rh = 50  # Relative humidity in %
    ta = T  # Air temperature in °C
    tr = T  # Mean radiant temperature in °C
    vel = 0.1  # Air velocity in m/s
    wme = 0  # External work in metabolic units

    # Determine clothing insulation (clo) and metabolic rate (met) based on time of day
    if day == 1:
        clo = 1
        met = 1.2
    else:
        clo = 2.5
        met = 0.8

    # Partial vapor pressure of water
    pa = rh * 10.0 * math.exp(16.6536 - 4030.183 / (ta + 235))

    # Convert clo to insulation in SI units
    icl = 0.155 * clo

    # Metabolic rate and work in watts per square meter
    m = met * 58.15
    w = wme * 58.15
    mw = m - w

    # Clothing area factor
    if icl <= 0.078:
        fcl = 1 + 1.29 * icl
    else:
        fcl = 1.05 + 0.645 * icl

    # Initial guess for clothing surface temperature
    taa = ta + 273  # Air temperature in Kelvin
    tra = tr + 273  # Radiant temperature in Kelvin
    tcla = taa + (35.5 - ta) / (3.5 * icl + 0.1)

    # Heat transfer coefficients and iterative calculation
    hcf = 12.1 * math.sqrt(vel)
    p1 = icl * fcl
    p2 = p1 * 3.96
    p3 = p1 * 100
    p4 = p1 * taa
    p5 = 308.7 - 0.028 * mw + (p2 * ((tra / 100.0) ** 4))
    xn = tcla / 100
    xf = tcla / 50
    eps = 0.0001

    # Iterate to solve for clothing surface temperature
    while abs(xn - xf) > eps:
        xf = (xf + xn) / 2
        hcn = 2.38 * abs(100.0 * xf - taa) ** 0.25
        hc = max(hcf, hcn)
        xn = (p5 + p4 * hc - p2 * (xf ** 4)) / (100.0 + p3 * hc)

    tcl = 100.0 * xn - 273  # Final clothing surface temperature in °C

    # Heat loss components
    hl1 = 3.05 * 0.001 * (5733.0 - (6.99 * mw) - pa)
    hl2 = 0.42 * (mw - 58.15) if mw > 58.15 else 0
    hl3 = 1.7 * 0.00001 * m * (5867.0 - pa)
    hl4 = 0.0014 * m * (34.0 - ta)
    hl5 = 3.96 * fcl * (xn ** 4 - (tra / 100.0) ** 4)
    hl6 = fcl * hc * (tcl - ta)

    # Predicted Mean Vote (PMV)
    ts = 0.303 * math.exp(-0.036 * m) + 0.028
    PMV = ts * (mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)

    # Percentage of People Dissatisfied (PPD)
    PPD = 100.0 - 95.0 * math.exp(-0.0353 * PMV ** 4 - 0.2179 * PMV ** 2)

    return PPD, PMV

 
    
# Calculate the advantage estimate
def Advantage_func(rewards, gamma):
    R = torch.zeros(1, 1).double()
    T = len(rewards)
    advantage = torch.zeros((T,1)).double()
    
    for i in reversed(range(len(rewards))):
        R = gamma * R + rewards[i]
        advantage[i] = R
    return advantage

class Dataset(data.Dataset):
    def __init__(self, states, actions, next_states, disturbance, rewards, targets):
        print("dataset init")
        self.states = states
        self.actions = actions
        self.next_states = next_states
        self.disturbance = disturbance
        self.rewards = rewards
        self.targets = targets
        #self.CC = CC
        #self.cc = cc

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):

        #print("targets shape", self.targets.shape)
        #print("disutbance shape", self.disturbance.shape)
        return self.states[index], self.actions[index], self.next_states[index], self.disturbance[index], self.rewards[index], self.targets[index]#, self.CC[index], self.cc[index]
    
class Replay_Memory():
    def __init__(self, memory_size=10):
        self.memory_size = memory_size
        self.len = 0
        self.rewards = []
        self.states = []
        self.n_states = []
        self.actions = []
        self.disturbance = []
        self.targets = []
        
        #self.CC = []
        #self.cc = []

    def sample_batch(self, batch_size):
        rand_idx = np.arange(-batch_size, 0, 1)
        batch_rewards = torch.stack([torch.tensor(self.rewards[i]) for i in rand_idx]).reshape(-1)
        batch_states = torch.stack([self.states[i] for i in rand_idx])
        batch_nStates = torch.stack([self.n_states[i] for i in rand_idx])
        batch_actions = torch.stack([self.actions[i] for i in rand_idx])
        batch_disturbance = torch.stack([self.disturbance[i] for i in rand_idx])
        batch_targets = torch.stack([self.targets[i] for i in rand_idx])
        
        #batch_CC = torch.stack([self.CC[i] for i in rand_idx])
        #batch_cc = torch.stack([self.cc[i] for i in rand_idx])
        # Flatten
        _, _, n_state =  batch_states.shape
        batch_states = batch_states.reshape(-1, n_state)
        batch_nStates = batch_nStates.reshape(-1, n_state)
        _, _, n_action =  batch_actions.shape
        batch_actions = batch_actions.reshape(-1, n_action)
        _, _, T, n_dist =  batch_disturbance.shape
        batch_disturbance = batch_disturbance.reshape(-1, T, n_dist)
        _, _, T, n_target =  batch_targets.shape
        batch_targets = batch_targets.reshape(-1, T, n_target)
        #_, _, T, n_tau, n_tau =  batch_CC.shape
        #batch_CC = batch_CC.reshape(-1, T, n_tau, n_tau)
        #batch_cc = batch_cc.reshape(-1, T, n_tau)
        return batch_states, batch_actions, batch_nStates, batch_disturbance, batch_rewards, batch_targets #, batch_CC, batch_cc

    def append(self, states, actions, next_states, rewards, dist, target_values):
        self.rewards.append(rewards)
        self.states.append(states)
        self.n_states.append(next_states)
        self.actions.append(actions)
        self.disturbance.append(dist)
        self.targets.append(target_values)
        #self.CC.append(CC)
        #self.cc.append(cc)
        self.len += 1
        
        if self.len > self.memory_size:
            self.len = self.memory_size
            self.rewards = self.rewards[-self.memory_size:]
            self.states = self.states[-self.memory_size:]
            self.actions = self.actions[-self.memory_size:]
            self.nStates = self.n_states[-self.memory_size:]
            self.disturbance = self.disturbance[-self.memory_size:]
            self.targets = self.targets[-self.memory_size:]
            #self.CC = self.CC[-self.memory_size:]
            #self.cc = self.cc[-self.memory_size:]

