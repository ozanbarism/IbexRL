import os
import sys
import gym
import gc 
from utils import make_dict, R_func
#turn off the warnings
import warnings
warnings.filterwarnings('ignore')

path_for_your_repo = "/Users/ozanbaris/Documents/GitHub/IbexRLExperiments/BoptestExperiment"  # Adjust this path based on your file structure
# these are respective paths that are automatically configured based on your repo path.
mpc_path = os.path.abspath(f"{path_for_your_repo}/IbexRL") 
agent_path = os.path.abspath(f"{path_for_your_repo}/IbexRL/agent")  
gnu_rl_path = f"{path_for_your_repo}/Gnu-RL" 

sys.path.insert(0, mpc_path)
from diff_mpc import mpc
from diff_mpc.mpc import QuadCost, LinDx

sys.path.insert(0, agent_path)
from numpy import genfromtxt
import numpy as np
import pickle
import pandas as pd
from scipy.linalg import expm
from numpy.linalg import inv


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# path for observations coming from the existing boptest controller.
path_for_observations= f"{gnu_rl_path}/observations.csv"
# boptest did not present actions as a part of the observations and that is why we had to extract them from another file.
path_for_actions= f"{gnu_rl_path}/results_tests_last_model_constant/results_tests_last_model_constant/results_sim_0.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# Define arguments directly
class Args:
    lr = 5e-3 #learning rate
    T = 24 # planning horizon
    step = 1800 #Timestep in in simulation, unit in seconds.
    O_hat_init = 1 # Initial guess for the hyperparameter for Balancing Comfort and Energy
    batch_size = 24 #Size of Mini-batch
    save_name = f'rl_lr'
    epoches = 50 # number of training epochs 
    lamda = 1 # Weight for state vs action priority in the objective function
    R_hat = 0.000001 # R_hat value, can be adjusted based on your model's requirements
    no_constraint= False # For ablation study, assigning True, will remove the constraints on parameter values
    pre_decided_init = False # For ablation study, assigning True, will provide chosen initial estimates. Assigning False will randomly select from predefined bounds.
    init = 1 # an integer for the torch seed. We vary this from 1 to 10 for our experiment. 
    s_t = 0.1 # linear control penalty coefficient. Rest of the values (O_hat and R_hat) are learned automatically so leaving this as a constant.

args = Args()


# Descriptive list of available observation & action names from BOPTEST
obs_name = [
    'reaPHeaPum_y',             # Heat pump electrical power (W)
    'reaCOP_y',                 # Heat pump COP (1)
    'reaTSetCoo_y',             # Zone operative temperature setpoint for cooling (K)
    'reaTSetHea_y',             # Zone operative temperature setpoint for heating (K)
    'reaTZon_y',                # Zone operative temperature (K)
    'weaSta_reaWeaHGloHor_y',   # Global horizontal solar irradiation measurement (W/m2)
    'weaSta_reaWeaTDryBul_y',   # Outside drybulb temperature measurement (K)
    'weaSta_reaWeaTWetBul_y',   # Wet bulb temperature measurement (K)
    'oveHeaPumY_u'              # Heat Pump Action (from actions_data.csv)
]

# Modify here: Based on your specific control problem and available BOPTEST data
state_name = ['reaTZon_y'] # Zone operative temperature

# Disturbances available from your configured BOPTEST observations
dist_name = [
    'weaSta_reaWeaTDryBul_y',   # Maps to "Outdoor Temp."
    'weaSta_reaWeaHGloHor_y'    # Maps to a "Solar Rad." component
]

# Control action - this is what the agent learns. It's your expert action from BOPTEST.
ctrl_name = ['oveHeaPumY_u'] # Directly use the heat pump action

# Target for the state variable (e.g., the setpoint for indoor temperature)
target_name = ['reaTSetHea_y'] # Using heating setpoint as the primary target.
                               # You could also use 'reaTSetCoo_y' or create a
                               # combined "effective setpoint" column during data preprocessing.

n_dist = len(dist_name) # Number of disturbances
n_state = len(state_name) # Number of state variables
n_ctrl = len(ctrl_name) # Number of control actions

time_step_seconds = args.step
T = args.T
step= args.step

# 1. Load observations from CSV and set timestamp index
observations_df = pd.read_csv(path_for_observations, parse_dates=['timestamp'], index_col='timestamp')

print("observations_df head: ", observations_df.head())
# 2. Load actions from another csv with 1 minute sampling rate. 
# Define the desired shape
rows = 2880
columns = 1
shape = (rows, columns)
# read actions_df from CSV file
actions_df = pd.read_csv(path_for_actions)

# Convert its 'oveHeaPumY_u' column to a numpy array
actions_df['datetime'] = pd.to_datetime(actions_df['datetime'])  # Ensure timestamp is in datetime format
actions_df.set_index('datetime', inplace=True)

# Resample from 30 seconds to 30 minutes and take the mean for resampling
actions_df_resampled = actions_df['oveHeaPumY_u'].resample('30T').mean()

# Convert the resampled series to a numpy array
processed_actions = actions_df_resampled.values

# Double check and print how many rows it has after resampling
print(f"Number of rows after resampling: {len(processed_actions)}")

# 4. Align lengths: observations_df has one more row, so drop its last row.
if len(observations_df) == len(processed_actions) + 1:
    observations_df = observations_df.iloc[:-1]
elif len(observations_df) != len(processed_actions):
    # Forcing alignment by truncating to the shortest if a different mismatch occurs.
    min_len = min(len(observations_df), len(processed_actions))
    observations_df = observations_df.iloc[:min_len]
    processed_actions = processed_actions[:min_len]


# 5. Create actions_df using the (now aligned) index from observations_df
actions_df = pd.DataFrame(processed_actions, index=observations_df.index, columns=ctrl_name)

#in the dataset convert the K to C for     'reaTSetCoo_y', 'reaTSetHea_y', 'reaTZon_y' 'weaSta_reaWeaTDryBul_y',
observations_df['weaSta_reaWeaTDryBul_y'] = observations_df['weaSta_reaWeaTDryBul_y'] - 273.15
observations_df['reaTSetCoo_y'] = observations_df['reaTSetCoo_y'] - 273.15
observations_df['reaTSetHea_y'] = observations_df['reaTSetHea_y'] - 273.15
observations_df['reaTZon_y'] = observations_df['reaTZon_y'] - 273.15
#convert the 'weaSta_reaWeaHGloHor_y' to kW/m2
observations_df['weaSta_reaWeaHGloHor_y'] = observations_df['weaSta_reaWeaHGloHor_y'] / 1000.0  # Convert W/m2 to kW/m2

# 6. Combine into a single dataset DataFrame
dataset = pd.concat([observations_df, actions_df], axis=1)

u_upper=1
u_lower=0


# Select the target column
target = dataset[target_name]

# Use the modified dist_name to select disturbance columns
disturbance = dataset[dist_name]

# First 30 days for training, next 15 days for testing, data is sampled every 30 minutes for two months. Compute the number of samples for 30 days
n_train = int(30 * 24 * 60 / (args.step / 60))
print("n_train: ", n_train)
n_test = int(15 * 24 * 60 / (args.step / 60)) # Ensure n_test fits within remaining data
print("n_test: ", n_test)
train_set = dataset[:n_train]
test_set = dataset[n_train:n_train + n_test]


class Learner():
    def __init__(self, n_state, n_ctrl, n_dist, disturbance, target, u_upper, u_lower):
        self.n_state = n_state
        self.n_ctrl = n_ctrl
        self.n_dist = n_dist
        self.disturbance = disturbance
        self.target = target

        if n_ctrl == 1:
            self.u_lower = torch.full((T, 1, n_ctrl), u_lower, dtype=torch.double)
            self.u_upper = torch.full((T, 1, n_ctrl), u_upper, dtype=torch.double)
        else:
            # Convert u_lower and u_upper to tensors and expand to match the desired shape
            self.u_lower = torch.tensor(u_lower, dtype=torch.double).expand(T, 1, n_ctrl)
            self.u_upper = torch.tensor(u_upper, dtype=torch.double).expand(T, 1, n_ctrl)


        if args.pre_decided_init:
            # Define learnable model parameters with requires_grad=True
            self.Rm = nn.Parameter(torch.tensor(1.00, dtype=torch.double), requires_grad=True)
            self.Rout = nn.Parameter(torch.tensor(1.50, dtype=torch.double), requires_grad=True)
            self.C = nn.Parameter(torch.tensor(5, dtype=torch.double), requires_grad=True)
            self.Ai = nn.Parameter(torch.tensor(0.5, dtype=torch.double), requires_grad=True)
            self.Tm = nn.Parameter(torch.tensor(21.0, dtype=torch.double), requires_grad=True)
            self.P_max = nn.Parameter(torch.tensor( 3, dtype=torch.double), requires_grad=True)
        else:
            bounds = { "Rm": (1e-2, 3), "Rout": (1e-2, 3), "C": (3, 15), "Tm": (19, 23), "Ai": (1e-3, 5), "P_max": (2, 4) } 
            # Initialize learnable model parameters with requires_grad=True
            self.Rm = nn.Parameter(torch.tensor(np.random.uniform(*bounds["Rm"]), dtype=torch.double), requires_grad=True)
            self.Rout = nn.Parameter(torch.tensor(np.random.uniform(*bounds["Rout"]), dtype=torch.double), requires_grad=True)
            self.C = nn.Parameter(torch.tensor(np.random.uniform(*bounds["C"]), dtype=torch.double), requires_grad=True)
            self.Ai = nn.Parameter(torch.tensor(np.random.uniform(*bounds["Ai"]), dtype=torch.double), requires_grad=True)
            self.Tm = nn.Parameter(torch.tensor(np.random.uniform(*bounds["Tm"]), dtype=torch.double), requires_grad=True)
            self.P_max = nn.Parameter(torch.tensor(np.random.uniform(*bounds["P_max"]), dtype=torch.double), requires_grad=True)


        
        self.O_hat= nn.Parameter(torch.tensor(args.eta, dtype=torch.double), requires_grad=True)
        # Initialize R_hat with ones and set specific  values as needed
        self.R_hat = nn.Parameter(torch.tensor(args.R_hat, dtype=torch.double), requires_grad=True)


        # Define the optimizer with all parameters that require gradients
        self.optimizer = optim.Adam(
            [self.Rm, self.Rout, self.C, self.Tm, self.Ai, self.O_hat, self.R_hat, self.P_max],
            lr=args.lr
        )


        # Initialize matrix placeholders to be recalculated in update_matrices
        self.Ac = None
        self.Bu = None
        self.Bd = None
        self.F = None
        self.G_u = None
        self.Bd_hat = None
        self.F_hat = None

    def update_matrices(self):

        # Calculate `Ac` based on Rm and Rout, ensuring it’s connected to autograd
        self.Ac = -1 / (self.C * args.step * self.Rm) - 1 / (self.C * args.step *self.Rout)
        self.Ac = self.Ac.unsqueeze(0).unsqueeze(1)  # Shape: (1, 1)

        # Define `Bu` based on the number of control actions, using autograd-friendly tensors
        if self.n_ctrl == 1:
            #self.Bu = torch.tensor([[self.P_max / (self.C *args.step)]], dtype=torch.double)
            bu_scalar_value = self.P_max / (self.C * args.step)
            self.Bu = bu_scalar_value.reshape(1, 1)
        elif self.n_ctrl == 3:
            self.Bu = torch.stack([1 / (self.C*args.step), self.eta_aux / (self.C * args.step), self.eta_aux / (self.C*args.step)], dim=0).unsqueeze(0)

        # Define `Bd` using learnable parameters directly
        self.Bd = torch.stack([1 / (self.C*args.step), 1 / (self.C*args.step), self.Ai / (args.step*self.C)], dim=0).unsqueeze(0)

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
        values = torch.cat([self.O_hat.unsqueeze(0), self.R_hat.unsqueeze(0)])
        self.C_hat = torch.diag(values)

    def Cost_function(self, cur_time):

        diag = torch.zeros(T, self.n_state + self.n_ctrl, dtype=torch.double)

        occupied_slice = round(self.target["reaTSetHea_y"][cur_time:cur_time + pd.Timedelta(seconds=(T - 1) * step)]) == 21
        occupied_np = np.array(occupied_slice)

        if len(occupied_np) < T:
            occupied_np = np.pad(occupied_np, (0, T - len(occupied_np)), 'edge')

        occupied_tensor = torch.tensor(occupied_np, dtype=torch.double).unsqueeze(1) # Shape (T, 1), contains 0.0 or 1.0
        unoccupied_weight = 0.0001

        eta_w_flag = (occupied_tensor * self.O_hat) + ((1 - occupied_tensor) * unoccupied_weight)

        diag[:, :self.n_state] = eta_w_flag # double assigned to double slice

        # Handle args.R_hat data type
        if torch.is_tensor(args.R_hat):
            diag[:, self.n_state:] = self.R_hat # NEW
        else:
            diag[:, self.n_state:] = self.R_hat # NEW

        self.C_cost = []
        for i in range(T):
            self.C_cost.append(torch.diag(diag[i])) # diag[i] is double, so torch.diag result is double
        self.C_cost = torch.stack(self.C_cost).unsqueeze(1) # T x 1 x (m+n) x (m+n), Double

        #print("C_cost shape:", self.C_cost.shape)
        x_target = self.target[cur_time : cur_time + pd.Timedelta(seconds=(T - 1) * step)]
        x_target = torch.tensor(np.array(x_target), dtype=torch.double)
        if len(x_target) < T:
            x_target = np.pad(x_target, ((0, T - len(x_target)), (0, 0)), 'edge')
        x_target = torch.tensor(x_target).double()
        
        # Make c explicitly dependent on O_hat
        c = torch.zeros(T, self.n_state + self.n_ctrl, dtype=torch.double)
        c[:, :self.n_state] = -eta_w_flag * x_target
        c[:, self.n_state:] = args.s_t  # Linear control penalty 
        c = c.unsqueeze(1)
        
        return self.C_cost, c

    def forward(self, x_init, C, c, cur_time):
        # Update matrices at each forward pass to reflect current parameter values
        self.update_matrices()

        # Prepare disturbances for prediction step
        dt = np.array(self.disturbance[cur_time : cur_time + pd.Timedelta(seconds=(T - 2) * step)])
        if len(dt) < T - 1:
            dt = np.pad(dt, ((0, T - 1 - len(dt)), (0, 0)), 'edge')
        dt = torch.tensor(dt).transpose(0, 1).double()
        
        # Repeat `Tm` and concatenate with disturbances
        Tm_tensor = self.Tm.expand(1, T - 1)
        dt = torch.cat((Tm_tensor, dt), dim=0)

        # Calculate `ft` based on `Bd_hat` and disturbances
        ft = torch.mm(self.Bd_hat, dt).transpose(0, 1).unsqueeze(1)

        # MPC function to optimize control actions over the planning horizon `T`
        F_hat_repeated = self.F_hat.repeat(T - 1, 1, 1, 1)

        # Extract the next T-1 COP values for the prediction horizon
        cop_values_raw = dataset.loc[cur_time : cur_time + pd.Timedelta(seconds=(T - 1) * step), 'reaCOP_y'].values


        # Check if we have at least T-1 values; if not, extend by repeating the last value
        if len(cop_values_raw) < T-1:
            last_value = cop_values_raw[-1] if len(cop_values_raw) > 0 else 1.0  # Use 1.0 as a fallback if empty
            cop_values_raw = np.pad(cop_values_raw, (0, T-1 - len(cop_values_raw)), 'edge')  # Repeat last value

        # Ensure exactly T-1 values are used
        cop_values_raw = cop_values_raw[:T-1]

        # Convert cop_values to a tensor
        cop_values = torch.tensor(cop_values_raw, dtype=torch.double).detach()

        # Reshape cop_values to (T-1, 1, 1) to align with F_hat_repeated for broadcasting
        cop_values = cop_values.view(T-1, 1, 1)

        # # Incorporate time-varying heat pump efficiency (COP) into the system dynamics.
        # The G_u matrix (the second column of F_hat) models the control input's effect.
        # We scale this effect by the predicted COP at each future timestep.
        F_hat_repeated[:, :, 0, 1] *= cop_values.squeeze(-1)  # Ensure it matches [T-1, 1, 1] structure


        x_pred, u_pred, _ = mpc.MPC(
            n_state=self.n_state,
            n_ctrl=self.n_ctrl,
            T=T,
            u_lower=self.u_lower,
            u_upper=self.u_upper,
            lqr_iter=20,
            verbose=0,
            exit_unconverged=False,
        )(x_init.double(), QuadCost(self.C_cost, c), LinDx(F_hat_repeated, ft))

        
        return x_pred[1, 0, :], u_pred[0, 0, :]
    def predict(self, x_init, action, cur_time):
        # Update matrices and calculate F_hat
        self.update_matrices()

        # Prepare disturbances for the prediction step
        dt = torch.tensor(np.array(self.disturbance.loc[cur_time])).unsqueeze(1).double()
        Tm_tensor = self.Tm.unsqueeze(0).unsqueeze(1)
        dt = torch.cat((Tm_tensor, dt), dim=0)

        # Calculate `ft` based on `Bd_hat` and disturbances
        ft = torch.mm(self.Bd_hat, dt)

        # Ensure x_init and action have the correct dimensions
        x_init = x_init.unsqueeze(0) if x_init.dim() == 1 else x_init
        action = action.unsqueeze(0) if action.dim() == 1 else action

        # Create an adjusted version of F_hat for this prediction step
        adjusted_F_hat = self.F_hat.clone()
        # Multiply the specific element in G_u matrix by the 'COP' value
        adjusted_F_hat[0, 1] *= dataset['reaCOP_y'][cur_time]

        # Concatenate initial state and action for next state prediction
        tau = torch.cat([x_init, action], dim=1)
        
        # Calculate the next state using adjusted_F_hat
        next_state = torch.mm(adjusted_F_hat, tau.T) + ft
        
        return next_state

    def update_parameters(self, x_true, u_true, x_pred, u_pred):
        state_loss =torch.mean((x_true.double() - x_pred) ** 2)
        action_loss = torch.mean((u_true.double() - u_pred) ** 2)
        traj_loss = state_loss + args.lamda * action_loss
        self.optimizer.zero_grad()
        traj_loss.backward(retain_graph=True)

        learnable_params = [self.Rm, self.Rout, self.C, self.Tm, self.Ai, self.O_hat, self.R_hat, self.P_max]
        torch.nn.utils.clip_grad_norm_(learnable_params, max_norm=1.0)

        self.optimizer.step()

        if not args.no_constraint:
            print("Clamping parameters to constraints")
            with torch.no_grad():
                self.C.data.clamp_(min=1)
                self.C.data.clamp_(max=15)
                self.Rm.data.clamp_(min=1e-6)
                self.Rout.data.clamp_(min=1e-6)
                self.Tm.data.clamp_(min=12)
                self.Tm.data.clamp_(max=25)
                self.Ai.data.clamp_(min=1e-9)
                self.Ai.data.clamp_(max=5)
                self.P_max.data.clamp_(min=1)
                self.P_max.data.clamp_(max=4)
        with torch.no_grad():
            self.O_hat.data.clamp_(min=0.001)
            self.R_hat.data.clamp_(min=1e-8)

        return state_loss.detach(), action_loss.detach()

        
def evaluate_performance(x_true, u_true, x_pred, u_pred):
    state_loss = torch.mean((x_true.double() - x_pred)**2)
    action_loss = torch.mean((u_true.double() - u_pred)**2)
    return state_loss, action_loss

def main():

    list_O_hat = [1, 0.1, 0.05, 0.005]
    list_lr = [5e-3, 1e-4, 5e-4, 1e-3]

    for O_hat_init in list_O_hat:
        for lr in list_lr:
            for init in [1,2,3,4,5, 42]:
                args.init = init
                lamda = O_hat_init # we use a lamda that is equal to O_hat_init but one might change it.

                if init == 42:
                    print("Using pre-decided initialization")
                    args.pre_decided_init = True
                else:
                    args.pre_decided_init = False

                torch.manual_seed(args.init)
                for cons_flag in [True, False]:
                    args.no_constraint = cons_flag

                    print(f"Running with eta={O_hat_init}, lr={lr}, lamda={lamda}, init={args.init}, no_constraint={args.no_constraint}")

                    args.lr = lr
                    args.lamda = lamda
                    args.O_hat_init = O_hat_init


                    if args.no_constraint:
                        dir = f'lr{args.lr}_eta{args.O_hat_init}_lamda{args.lamda}_noconstraint'
                    else:
                        dir = f'lr{args.lr}_eta{args.O_hat_init}_lamda{args.lamda}_constraint'

                    if args.pre_decided_init:
                        dir = f"{dir}_initdecided"
                    else:
                        dir = f"{dir}_init{args.init}"

                    #check if the epoch 49 exists, if it does, continue
                    if os.path.exists(os.path.join(dir, 'Imit_rl_lr_49.pkl')):
                        print(f"Skipping existing directory: {dir}")
                        continue

                    # Initialize a dictionary to store parameter values for each epoch
                    parameters_per_epoch = {
                        "Epoch": [],
                        "Rm": [],
                        "Rout": [],
                        "Capacitance": [],
                        "Tm": [],
                        "Ai": [],
                        "P_max": [],
                        "O_hat": [],
                        "R_hat": [],  # For first component of R_hat
                    }


                    if not os.path.exists(dir):
                        os.mkdir(dir)
                    numOfEpoches = args.epoches
                    
                    timeStamp = []
                    record_name = ["Learner nState", "Expert nState"]

                    # Add column names for actions dynamically based on the number of actions (n_ctrl)
                    if n_ctrl > 1:
                        record_name += [f"Learner action {i+1}" for i in range(n_ctrl)]
                        record_name += [f"Expert action {i+1}" for i in range(n_ctrl)]
                    else:
                        record_name += ["Learner action", "Expert action"]
                    losses = []
                    losses_name = ["train_state_loss", "train_action_loss", "val_state_loss", "val_action_loss"]
                    
                    learner = Learner(n_state, n_ctrl, n_dist, disturbance, target, u_upper, u_lower)
                    
                    for epoch in range(numOfEpoches):
                        x_true = []
                        u_true = []
                        x_pred = []
                        u_pred = []
                        
                        train_state_loss = []
                        train_action_loss = []
                        for i in range(n_train): # By number of entries in the historical data
                            idx = np.random.randint(n_train)
                            cur_time = train_set.index[idx]
                            expert_moves = train_set[cur_time:cur_time+pd.Timedelta(seconds = step)]
                            if len(expert_moves)<2:
                                print(cur_time)
                                continue
                            
                            expert_state = torch.tensor(expert_moves[state_name].values).reshape(-1, n_state) # 2 x n_state
                            expert_action = torch.tensor(expert_moves[ctrl_name].values).reshape(-1, n_ctrl) # 2 x n_ctrl
                            x_true.append(expert_state[-1])
                            u_true.append(expert_action[0])
                            #print('u_true', u_true)

                            obs = train_set.loc[cur_time]
                            x_init = torch.tensor(np.array([obs[name] for name in state_name])).unsqueeze(0).double()
                            # n_batch x n_state, i.e. 1 x n_state
                            C, c = learner.Cost_function(cur_time)
                            learner_state, learner_action = learner.forward(x_init, C, c, cur_time)
                            
                            # Predict next state based on expert's action
                            next_state = learner.predict(x_init.squeeze(0), expert_action[0], cur_time)
                            x_pred.append(next_state)
                            u_pred.append(learner_action)
                            
                            if (i % args.batch_size == 0) & (i>0):
                                x_true = torch.stack(x_true).reshape(-1, n_state)
                                u_true = torch.stack(u_true).reshape(-1, n_ctrl)
                                x_pred = torch.stack(x_pred).reshape(-1, n_state)
                                u_pred = torch.stack(u_pred).reshape(-1, n_ctrl)
                                b_state_loss, b_action_loss = learner.update_parameters(x_true, u_true, x_pred, u_pred)
                                train_state_loss.append(b_state_loss)
                                train_action_loss.append(b_action_loss)
                                #print("at epoch {0}, batch {1}, the state loss is {2} and the action loss is {3}".format(epoch, i//args.batch_size, b_state_loss, b_action_loss))
                                x_true = []
                                u_true = []
                                x_pred = []
                                u_pred = []

                        # Evaluate performance at the end of each epoch
                        x_true = []
                        u_true = []
                        x_pred = []
                        u_pred = []
                        timeStamp = []
                        for idx in range(n_test):
                            cur_time = test_set.index[idx]
                            expert_moves = test_set[cur_time:cur_time+pd.Timedelta(seconds = step)]
                            if len(expert_moves)<2:
                                print(cur_time)
                                continue
                            expert_state = torch.tensor(expert_moves[state_name].values).reshape(-1, n_state) # 2 x n_state
                            expert_action = torch.tensor(expert_moves[ctrl_name].values).reshape(-1, n_ctrl) # 2 x n_ctrl
                            x_true.append(expert_state[-1])
                            u_true.append(expert_action[0])
                            
                            timeStamp.append(cur_time+pd.Timedelta(seconds = step))
                            
                            obs = test_set.loc[cur_time]
                            x_init = torch.tensor(np.array([obs[name] for name in state_name])).unsqueeze(0) # 1 x n_state
                            C, c = learner.Cost_function(cur_time)
                            learner_state, learner_action = learner.forward(x_init, C, c, cur_time)
                            next_state = learner.predict(x_init.squeeze(0), expert_action[0], cur_time)
                            #print("expert action", expert_action[0], "learner action", learner_action, "next state", next_state, "expert state", expert_state[-1])

                            x_pred.append(next_state.detach())
                            u_pred.append(learner_action.detach())
                            
                        x_true = torch.stack(x_true).reshape(-1, n_state)
                        u_true = torch.stack(u_true).reshape(-1, n_ctrl)
                        x_pred = torch.stack(x_pred).reshape(-1, n_state)
                        u_pred = torch.stack(u_pred).reshape(-1, n_ctrl)
                        val_state_loss, val_action_loss = evaluate_performance(x_true, u_true, x_pred, u_pred)
                        print("At Epoch {0}, the loss from the state is {1} and from the action is {2}".format(epoch, val_state_loss, val_action_loss))
                        losses.append((np.mean(train_state_loss), np.mean(train_action_loss), val_state_loss, val_action_loss))
                    
                        record = pd.DataFrame(torch.cat((x_pred, x_true, u_pred, u_true), dim = 1).numpy(), index = np.array(timeStamp), columns = record_name)
                        record_df = pd.DataFrame(np.array(record), index = np.array(timeStamp), columns = record_name)
                        record_df.to_pickle("{}/Imit_{}_{}.pkl".format(dir,args.save_name, epoch))

                        # Save weights
                        F_hat = learner.F_hat.detach().numpy()
                        Bd_hat = learner.Bd_hat.detach().numpy()
                        C_hat = learner.C_hat.detach().numpy()

                        weight_dir = f"{dir}/weights"
                        if not os.path.exists(weight_dir):
                            os.mkdir(weight_dir)

                        np.save("{}/weights/F-{}.npy".format(dir,epoch), F_hat)
                        np.save("{}/weights/Bd-{}.npy".format(dir,epoch), Bd_hat)
                        np.save("{}/weights/C_hat-{}.npy".format(dir,epoch), C_hat)

                        #parameter directory check
                        if not os.path.exists(f"{dir}/parameters"):
                            os.mkdir(f"{dir}/parameters")



                        #save parameters fitted
                        Rm= learner.Rm.detach().numpy()
                        Rout= learner.Rout.detach().numpy()
                        Capacitance= learner.C.detach().numpy()
                        Tm= learner.Tm.detach().numpy()
                        Ai= learner.Ai.detach().numpy()
                        P_max = learner.P_max.detach().numpy()
                        O_hat= learner.O_hat.detach().numpy()
                        R_hat= learner.R_hat.detach().numpy()
                        np.save("{}/parameters/Rm-{}.npy".format(dir,epoch), Rm)
                        np.save("{}/parameters/Rout-{}.npy".format(dir,epoch), Rout)
                        np.save("{}/parameters/C-{}.npy".format(dir,epoch), Capacitance)
                        np.save("{}/parameters/Tm-{}.npy".format(dir,epoch), Tm)
                        np.save("{}/parameters/Ai-{}.npy".format(dir,epoch), Ai)
                        np.save("{}/parameters/P_max-{}.npy".format(dir,epoch), P_max)
                        np.save("{}/parameters/O-{}.npy".format(dir,epoch), O_hat)
                        np.save("{}/parameters/R-{}.npy".format(dir,epoch), R_hat)


                        # Add this inside the loop after parameter updates
                        parameters_per_epoch["Epoch"].append(epoch)
                        parameters_per_epoch["Rm"].append(learner.Rm.detach().item())
                        parameters_per_epoch["Rout"].append(learner.Rout.detach().item())
                        parameters_per_epoch["Capacitance"].append(learner.C.detach().item())
                        parameters_per_epoch["Tm"].append(learner.Tm.detach().item())
                        parameters_per_epoch["Ai"].append(learner.Ai.detach().item())
                        parameters_per_epoch["P_max"].append(learner.P_max.detach().item())
                        parameters_per_epoch["O_hat"].append(learner.O_hat.detach().item())
                        parameters_per_epoch["R_hat"].append(learner.R_hat.detach().item())


                    # After the training loop, save the collected parameters as a CSV
                    output_file = f"{dir}/parameters/parameters_per_epoch.csv"
                    parameters_df = pd.DataFrame(parameters_per_epoch)
                    parameters_df.to_csv(output_file, index=False)
                    print(f"Saved parameter values for all epochs to {output_file}")


                    # Save losses at each epoch
                    losses_df = pd.DataFrame(np.array(losses), index = np.arange(numOfEpoches), columns = losses_name)
                    losses_df.to_pickle(f"{dir}/Imit_loss_"+args.save_name+".pkl")
                    print(f"Saved losses to {dir}/Imit_loss_{args.save_name}.pkl")    
                    print(f"Cleaning up memory after run with eta={O_hat}, lr={lr}, lamda={lamda}, init={args.init}, no_constraint={args.no_constraint}...")
                    del learner
                    del losses
                    del parameters_per_epoch

                    gc.collect()  # Trigger garbage collection
                    if DEVICE == "cuda":
                        torch.cuda.empty_cache()  # Clear PyTorch's CUDA cache
                    print("Memory cleanup complete for this run.")

if __name__ == "__main__":
    main()
    print("Training completed and results saved.")