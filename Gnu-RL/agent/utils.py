# Helper Functions
import numpy as np
import torch
import torch.utils.data as data
import torch

def make_dict(obs_name, obs):
    zipbObj = zip(obs_name, obs)
    return dict(zipbObj)

def R_func(obs_dict, action, eta_occ):
    reaTSetHea_y = round(obs_dict['reaTSetHea_y'])
    occupied = (reaTSetHea_y == 21)
    print("reaTSetHea_y value:", obs_dict['reaTSetHea_y'], "Type:", type(obs_dict['reaTSetHea_y']))
    print("reaTSetHea_y rounded:", reaTSetHea_y, "Occupied:", occupied, "coeff", eta_occ[int(occupied)])
    reward = - 0.5 * eta_occ[int(occupied)] * (obs_dict['reaTZon_y'] - obs_dict['reaTSetHea_y'])**2 - 0.1*action
    return reward.item()


def R_func_neg_cost(obs_dict, action, eta):
    """
    Calculates a reward that is the negative of the cost defined by
    the logic in the provided Cost_function.

    The cost J for a single step is assumed to be:
    J = 0.5 * x^T * C_xx * x + 0.5 * u^T * C_uu * u + c_x^T * x + c_u^T * u
    where:
        C_xx is diagonal with 'eta' for states.
        C_uu is diagonal with 0.001 for controls.
        c_x = -eta * x_target for states.
        c_u is a vector of ones for controls.

    Args:
        obs_dict (dict): Dictionary containing observations.
                         Expected keys:
                         'reaTZon_y': Current state (x).
                         'reaTSetHea_y': Target state (x_target).
        action (array-like): Control action (u).
        eta (float): Weight factor, corresponding to args.eta in Cost_function.

    Returns:
        torch.Tensor: The calculated reward (a scalar tensor).
    """

    # --- MODIFICATION START: Convert inputs to PyTorch tensors ---
    # Determine a target device (e.g., from 'action' if it's already a tensor, or default to CPU)
    # This helps ensure all tensors are on the same device.
    # If your 'DEVICE' constant from Cost_function is accessible here, you should use it.
    # For now, we'll try to infer or use None (which usually means CPU for new tensors
    # unless torch.get_default_device() is set otherwise).
    _device = None
    if isinstance(action, torch.Tensor):
        _device = action.device
    elif isinstance(obs_dict.get('reaTZon_y'), torch.Tensor): # Use .get() for safety
        _device = obs_dict['reaTZon_y'].device
    elif isinstance(obs_dict.get('reaTSetHea_y'), torch.Tensor):
        _device = obs_dict['reaTSetHea_y'].device

    # Convert inputs to PyTorch tensors with dtype=torch.double
    # torch.as_tensor will avoid a copy if the input is already a tensor
    # of the correct type and device (for NumPy it shares memory if compatible).
    current_state = torch.as_tensor(obs_dict['reaTZon_y'], dtype=torch.double, device=_device)
    target_state = torch.as_tensor(obs_dict['reaTSetHea_y'], dtype=torch.double, device=_device)
    action = torch.as_tensor(action, dtype=torch.double, device=_device)

    # 'eta' is likely a Python float. PyTorch can handle multiplication
    # of a tensor by a Python scalar. If eta itself needs to be a tensor:
    # eta_tensor = torch.as_tensor(eta, dtype=torch.double, device=_device)
    # Then use eta_tensor in calculations. For now, assuming scalar eta is fine.
    # --- MODIFICATION END ---

    # Cost term: 0.5 * x^T * C_xx * x
    # C_xx is diagonal with 'eta'. So, 0.5 * eta * sum(current_state_i^2)
    cost_x_quad = 0.5 * eta * torch.sum(current_state**2)

    # Cost term: 0.5 * u^T * C_uu * u
    # C_uu is diagonal with 0.001. So, 0.5 * 0.001 * sum(action_j^2)
    cost_u_quad = 0.5 * 1000 * torch.sum(action**2)

    # Cost term: c_x^T * x
    # c_x = -eta * x_target. So, -eta * sum(current_state_i * target_state_i)
    cost_x_linear = -eta * torch.sum(current_state * target_state) # Element-wise product then sum

    # Cost term: c_u^T * u
    # c_u is a vector of ones. So, sum(action_j)
    # This term comes from `c[:, n_state:] = 1` in the Cost_function
    cost_u_linear = torch.sum(action)

    total_cost = cost_x_quad + cost_u_quad + cost_x_linear + cost_u_linear

    print("state:", current_state, "target_state:", target_state, "action:", action)
    print("cost_x_quad:", cost_x_quad.item())
    print("cost_u_quad:", cost_u_quad.item())
    print("cost_x_linear:", cost_x_linear.item())
    print("cost_u_linear:", cost_u_linear.item())
    print("reward:", -total_cost.item())


    reward = -total_cost

    return reward # .item() if you need a Python scalar, but usually RL frameworks handle tensor rewards
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
    def __init__(self, states, actions, next_states, disturbance, rewards, old_logprobs, CC, cc):
        self.states = states
        self.actions = actions
        self.next_states = next_states
        self.disturbance = disturbance
        self.rewards = rewards
        self.old_logprobs = old_logprobs
        self.CC = CC
        self.cc = cc

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        return self.states[index], self.actions[index], self.next_states[index], self.disturbance[index], self.rewards[index], self.old_logprobs[index], self.CC[index], self.cc[index]
    
class Replay_Memory():
    def __init__(self, memory_size=10):
        self.memory_size = memory_size
        self.len = 0
        self.rewards = []
        self.states = []
        self.n_states = []
        self.log_probs = []
        self.actions = []
        self.disturbance = []
        self.CC = []
        self.cc = []

    def sample_batch(self, batch_size):
        rand_idx = np.arange(-batch_size, 0, 1)
        batch_rewards = torch.stack([self.rewards[i] for i in rand_idx]).reshape(-1)
        batch_states = torch.stack([self.states[i] for i in rand_idx])
        batch_nStates = torch.stack([self.n_states[i] for i in rand_idx])
        batch_actions = torch.stack([self.actions[i] for i in rand_idx])
        batch_logprobs = torch.stack([self.log_probs[i] for i in rand_idx]).reshape(-1)
        batch_disturbance = torch.stack([self.disturbance[i] for i in rand_idx])
        batch_CC = torch.stack([self.CC[i] for i in rand_idx])
        batch_cc = torch.stack([self.cc[i] for i in rand_idx])
        # Flatten
        _, _, n_state =  batch_states.shape
        batch_states = batch_states.reshape(-1, n_state)
        batch_nStates = batch_nStates.reshape(-1, n_state)
        _, _, n_action =  batch_actions.shape
        batch_actions = batch_actions.reshape(-1, n_action)
        _, _, T, n_dist =  batch_disturbance.shape
        batch_disturbance = batch_disturbance.reshape(-1, T, n_dist)
        _, _, T, n_tau, n_tau =  batch_CC.shape
        batch_CC = batch_CC.reshape(-1, T, n_tau, n_tau)
        batch_cc = batch_cc.reshape(-1, T, n_tau)
        return batch_states, batch_actions, batch_nStates, batch_disturbance, batch_rewards, batch_logprobs, batch_CC, batch_cc

    def append(self, states, actions, next_states, rewards, log_probs, dist, CC, cc):
        self.rewards.append(rewards)
        self.states.append(states)
        self.n_states.append(next_states)
        self.log_probs.append(log_probs)
        self.actions.append(actions)
        self.disturbance.append(dist)
        self.CC.append(CC)
        self.cc.append(cc)
        self.len += 1
        
        if self.len > self.memory_size:
            self.len = self.memory_size
            self.rewards = self.rewards[-self.memory_size:]
            self.states = self.states[-self.memory_size:]
            self.log_probs = self.log_probs[-self.memory_size:]
            self.actions = self.actions[-self.memory_size:]
            self.nStates = self.n_states[-self.memory_size:]
            self.disturbance = self.disturbance[-self.memory_size:]
            self.CC = self.CC[-self.memory_size:]
            self.cc = self.cc[-self.memory_size:]

