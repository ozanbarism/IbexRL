import os
import sys

import gym
import warnings
warnings.filterwarnings("ignore", message="indexing with dtype torch.uint8 is now deprecated")
# Assign mpc_path to be the file path where mpc.torch is located.
mpc_path = os.path.abspath(os.path.join(__file__, '..', '..' ))
sys.path.insert(0, mpc_path)

import argparse
from numpy import genfromtxt
import numpy as np
import pickle
import pandas as pd

from diff_mpc import mpc
from diff_mpc.mpc import QuadCost, LinDx

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import make_dict, R_func



class Learner():
    def __init__(self, n_state, n_ctrl, n_dist, disturbance, target, u_upper, u_lower):
        self.n_state = n_state
        self.n_ctrl = n_ctrl
        self.n_dist = n_dist
        self.disturbance = disturbance
        self.target = target
        
        # My Initial Guess
        self.F_hat = torch.ones((self.n_state, self.n_state+self.n_ctrl))
        self.F_hat[0, 0] = F1_init
        self.F_hat[0, 1] = F2_init
        self.F_hat = self.F_hat.double().requires_grad_()
        
        self.Bd_hat = np.random.rand(self.n_state, self.n_dist)
        self.Bd_hat = torch.tensor(self.Bd_hat).requires_grad_()
        
        self.optimizer = optim.Adam([self.F_hat, self.Bd_hat], lr=args.lr)
    
        self.u_lower = u_lower * torch.ones(T, 1, n_ctrl).double()
        self.u_upper = u_upper * torch.ones(T, 1, n_ctrl).double()
    
    def Cost_function(self, cur_time):
        diag = torch.zeros(T, self.n_state + self.n_ctrl)
        occupied = self.target["reaTSetHea_y"][cur_time:cur_time + pd.Timedelta(seconds = (T-1) * step)] == 21
        occupied = np.array(occupied)
        #print("Occupied: ", occupied)
        #print("target: ", self.target["reaTSetHea_y"][cur_time:cur_time + pd.Timedelta(seconds = (T-1) * step)])
        if len(occupied)<T:
            occupied = np.pad(occupied, ((0, T-len(occupied)), ), 'edge')
        eta_w_flag = torch.tensor([eta_occ[int(flag)] for flag in occupied]).unsqueeze(1).double() # Tx1

        diag[:, :self.n_state] = eta_w_flag
        diag[:, self.n_state:] = args.R_hat
        #print("diag: ", diag)
        C = []
        for i in range(T):
            C.append(torch.diag(diag[i]))
        C = torch.stack(C).unsqueeze(1) # T x 1 x (m+n) x (m+n)
        
        x_target = self.target[cur_time : cur_time + pd.Timedelta(seconds = (T-1) * step)] # in pd.Series
        x_target = np.array(x_target)
        if len(x_target)<T:
            x_target = np.pad(x_target, ((0, T-len(x_target)), (0, 0)), 'edge')
        x_target = torch.tensor(x_target)
        
        c = torch.zeros(T, self.n_state+self.n_ctrl) # T x (m+n)
        c[:, :n_state] = -eta_w_flag*x_target
        c[:, n_state:] = 0.1 # L1-norm now! Check
        c = c.unsqueeze(1) # T x 1 x (m+n)
        return C, c
    
    
    def forward(self, x_init, C, c, cur_time):
        dt = np.array(self.disturbance[cur_time : cur_time + pd.Timedelta(seconds = (T-2) * step)]) # T-1 x n_dist
        if len(dt)<T-1:
            dt = np.pad(dt, ((0, T-1-len(dt)), (0, 0)), 'edge')
        dt = torch.tensor(dt).transpose(0, 1) # n_dist x T-1
        
        ft = torch.mm(self.Bd_hat, dt).transpose(0, 1) # T-1 x n_state
        ft = ft.unsqueeze(1) # T-1 x 1 x n_state
        
        x_pred, u_pred, _ = mpc.MPC(n_state=self.n_state,
                                    n_ctrl=self.n_ctrl,
                                    T=T,
                                    u_lower = self.u_lower,
                                    u_upper = self.u_upper,
                                    lqr_iter=20,
                                    verbose=0,
                                    exit_unconverged=False,
                                    )(x_init, QuadCost(C.double(), c.double()),
                                      LinDx(self.F_hat.repeat(T-1, 1, 1, 1),  ft))
        
        return x_pred[1, 0, :], u_pred[0, 0, :] # Dim.
    
    def predict(self, x_init, action, cur_time):
        dt = np.array(self.disturbance.loc[cur_time]) # n_dist
        dt = torch.tensor(dt).unsqueeze(1) # n_dist x 1
        ft = torch.mm(self.Bd_hat, dt) # n_state x 1
        tau = torch.stack([x_init, action]) # (n_state + n_ctrl) x 1
        next_state  = torch.mm(self.F_hat, tau) + ft # n_state x 1
        #print("for tau {}, ft {} the next state is {}".format(tau, ft, next_state))
        return next_state
                                    
    def update_parameters(self, x_true, u_true, x_pred, u_pred):
        # Every thing in T x Dim.
        state_loss = torch.mean((x_true.double() - x_pred)**2)
        action_loss = torch.mean((u_true.double() - u_pred)**2)
        
        # Note: args.eta balances the importance between predicting states and predicting actions
        traj_loss = args.lamda*state_loss + action_loss
        print("From state {}, From action {}".format(state_loss, action_loss))
        self.optimizer.zero_grad()
        traj_loss.backward()
        self.optimizer.step()
        print("F_hat",self.F_hat)
        print("Bd_hat",self.Bd_hat)
        return state_loss.detach(), action_loss.detach()
        
def evaluate_performance(x_true, u_true, x_pred, u_pred):
    state_loss = torch.mean((x_true.double() - x_pred)**2)
    action_loss = torch.mean((u_true.double() - u_pred)**2)
    return state_loss, action_loss



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE


class Args:
    seed = 42
    lr = 5e-3 #learning rate
    T = 24 # planning horizon
    step = 1800 #Timestep in in simulation, unit in seconds.
    eta = 1#Hyper Parameter for Balancing Comfort and Energy
    batch_size = 24 #Size of Mini-batch
    epoches=50
    lamda = 1
    save_name = 'gnu-rl'
    R_hat=0.000001

args= Args()
torch.manual_seed(args.seed)

F1_init = 0.8
F2_init = 1

# Descriptive list of available observation & action names from BOPTEST
# (This list is mostly for documentation/reference within your script)
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
observations_df = pd.read_csv('observations.csv', parse_dates=['timestamp'], index_col='timestamp')

print("observations_df head: ", observations_df.head())

# Define the desired shape
rows = 2880
columns = 1
shape = (rows, columns)
# read actions_df from CSV file
actions_df = pd.read_csv('./results_tests_last_model_constant/results_sim_0.csv')

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
    # This case should not happen based on the problem description,
    # but as a minimal safety for unexpected length mismatches other than +1.
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


# 6. Combine into a single dataset DataFrame
dataset = pd.concat([observations_df, actions_df], axis=1)
print("dataset head: ", dataset.head())
# 7. Extract target and disturbance
target = dataset[target_name]
disturbance = dataset[dist_name]
print("disturbance unnormalized: \n", disturbance.head())
# 8. Min-Max Normalization for disturbances (if n_dist > 0)
if n_dist > 0 and not disturbance.empty:
    disturbance_min = disturbance.min()
    disturbance_range = disturbance.max() - disturbance_min
    disturbance_range[disturbance_range == 0] = 1e-6 # Avoid division by zero with a small epsilon
    disturbance = (disturbance - disturbance_min) / disturbance_range

#save the normalization parameters
disturbance_params = {
    'min': disturbance_min,
    'range': disturbance_range
}
with open('disturbance_params.pkl', 'wb') as f:
    pickle.dump(disturbance_params, f)

print("first 5 rows of disturbance after normalization: \n", disturbance.head())


print("Normalization Done.")


# First 30 days for training, next 15 days for testing, data is sampled every 30 minutes for two months. Compute the number of samples for 30 days
n_train = int(30 * 24 * 60 / (args.step / 60))
print("n_train: ", n_train)
n_test = int(15 * 24 * 60 / (args.step / 60)) # Ensure n_test fits within remaining data
print("n_test: ", n_test)
train_set = dataset[:n_train]
test_set = dataset[n_train:n_train + n_test]

if len(train_set) > 0 and len(test_set) > 0:
    print("first and last day of training set: ", train_set.index[0], train_set.index[-1])
    print("first and last day of testing set: ", test_set.index[0], test_set.index[-1])
else:
    print("Error: Insufficient data for the specified train-test split.")


list_eta = [0.1, 1,  0.01]#[1, 0.1, 0.05, 0.005]
list_lr = [0.001, 0.005,0.0001]#[5e-3, 1e-4, 5e-4, 1e-3]
list_lamda=[0.1]# [1]#, 10,100]
for eta in list_eta:
    for lr in list_lr:
        for lamda in list_lamda:
            lamda=eta
            # Define arguments directly

            args.lr = lr
            args.lamda = lamda
            args.eta = eta
       
            eta_occ = [0.0001, args.eta]

            dir = f'results_boptest_{args.eta}_{args.lr}_{args.lamda}'
            if not os.path.exists(dir):
                os.mkdir(dir)

            numOfEpoches = args.epoches

            timeStamp = []
            record_name =["Learner nState", "Expert nState", "Learner action", "Expert action"]
            losses = []
            losses_name = ["train_state_loss", "train_action_loss", "val_state_loss", "val_action_loss"]

            # Initialize the learner
            u_upper = 1
            u_lower = 0
            print("initializing learner with eta {}, lr {}, lambda {}".format(args.eta, args.lr, args.lamda))
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

                    obs = train_set.loc[cur_time]
                    x_init = torch.tensor(np.array([obs[name] for name in state_name])).unsqueeze(0) # n_batch x n_state, i.e. 1 x n_state
                    C, c = learner.Cost_function(cur_time)
                    learner_state, learner_action = learner.forward(x_init, C, c, cur_time)
                    
                    #once in every 10 iterations, print the current state and action
                    if i % 10 == 0:
                        print("Learner Action {}, Expert Action {}".format(learner_action.item(), expert_action[0]))

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

                #check if the directory exists
                if not os.path.exists(dir):
                    os.makedirs(dir)
                record_df.to_pickle("{}/Imit_{}_{}.pkl".format(dir,args.save_name, epoch))
                
                # Save weights
                F_hat = learner.F_hat.detach().numpy()
                Bd_hat = learner.Bd_hat.detach().numpy()
                #check if the directory exists
                if not os.path.exists("{}/weights".format(dir)):
                    os.makedirs("{}/weights".format(dir))
                np.save("{}/weights/F-{}.npy".format(dir,epoch), F_hat)
                np.save("{}/weights/Bd-{}.npy".format(dir,epoch), Bd_hat)
                
            # Save losses at each epoch
            losses_df = pd.DataFrame(np.array(losses), index = np.arange(numOfEpoches), columns = losses_name)
            losses_df.to_pickle(f"{dir}/Imit_loss_"+args.save_name+".pkl")
