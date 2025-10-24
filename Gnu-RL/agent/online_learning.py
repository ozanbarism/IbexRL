import os
import sys

import gym

#UPDATE THESE PATHS TO YOUR LOCAL GIT REPOSITORY
BOPTEST_GYM_PATH = '/Users/ozanbaris/Documents/GitHub/IbexRLSimulation/Gnu-RL/project1-boptest-gym'
sys.path.append(BOPTEST_GYM_PATH)
from boptestGymEnv import BoptestGymEnv


# Assign mpc_path to be the file path where mpc.torch is located.
mpc_path = os.path.abspath(os.path.join(__file__,'..', '..'))
sys.path.insert(0, mpc_path)

from diff_mpc import mpc
from diff_mpc.mpc import QuadCost, LinDx

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

from utils import make_dict, R_func, Advantage_func, Replay_Memory, Dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



class PPO():
    def __init__(self, memory, T, n_ctrl, n_state, target, disturbance, eta, u_upper, u_lower, clip_param = 0.1, F_hat = None, Bd_hat = None):
        self.memory = memory
        self.clip_param = clip_param
        
        self.T = T
        self.step = args.step
        self.n_ctrl = n_ctrl
        self.n_state = n_state
        self.eta = eta
        
        self.target = target
        self.dist = disturbance
        self.n_dist = self.dist.shape[1]
        
        if F_hat is not None:
            print(f"Loading pretrained F_hat from IL (shape: {F_hat.shape}). Expected: ({self.n_state}, {self.n_state + self.n_ctrl})")
            self.F_hat = torch.tensor(F_hat, dtype=torch.double, device=DEVICE).requires_grad_()

        if Bd_hat is not None:
            print(f"Loading pretrained Bd_hat from IL (shape: {Bd_hat.shape}). Expected: ({self.n_state}, {self.n_dist})")
            self.Bd_hat = torch.tensor(Bd_hat, dtype=torch.double, device=DEVICE).requires_grad_()

        self.Bd_hat = torch.tensor(self.Bd_hat).requires_grad_()
        print(self.Bd_hat)
        
        self.Bd_hat_old = self.Bd_hat.detach().clone()
        self.F_hat_old = self.F_hat.detach().clone()
        
        self.optimizer = optim.RMSprop([self.F_hat, self.Bd_hat], lr=args.lr)
        
        self.u_lower = u_lower * torch.ones(n_ctrl).double()
        self.u_upper = u_upper * torch.ones(n_ctrl).double()
    
    # Use the "current" flag to indicate which set of parameters to use
    def forward(self, x_init, ft, C, c, current = True, n_iters=20):
        T, n_batch, n_dist = ft.shape
        if current == True:
            F_hat = self.F_hat
            Bd_hat = self.Bd_hat
        else:
            F_hat = self.F_hat_old
            Bd_hat = self.Bd_hat_old
 
        x_lqr, u_lqr, objs_lqr = mpc.MPC(n_state=self.n_state,
                                         n_ctrl=self.n_ctrl,
                                         T=self.T,
                                         u_lower= self.u_lower.repeat(self.T, n_batch, 1),
                                         u_upper= self.u_upper.repeat(self.T, n_batch, 1),
                                         lqr_iter=n_iters,
                                         backprop = True,
                                         verbose=0,
                                         exit_unconverged=False,
                                         )(x_init.double(), QuadCost(C.double(), c.double()),
                                           LinDx(F_hat.repeat(self.T-1, n_batch, 1, 1), ft.double()))
        return x_lqr, u_lqr

    def select_action(self, mu, sigma):
        if self.n_ctrl > 1:
            sigma_sq = torch.ones(mu.size()).double() * sigma**2
            dist = MultivariateNormal(mu, torch.diag(sigma_sq.squeeze()).unsqueeze(0))
        else:
            dist = Normal(mu, torch.ones_like(mu)*sigma)    
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def evaluate_action(self, mu, actions, sigma):
        n_batch = len(mu)
        if self.n_ctrl > 1:
            cov = torch.eye(self.n_ctrl).double() * sigma**2
            cov = cov.repeat(n_batch, 1, 1)
            dist = MultivariateNormal(mu, cov)
        else:
            dist = Normal(mu, torch.ones_like(mu)*sigma)
        log_prob = dist.log_prob(actions.double())
        entropy = dist.entropy()
        return log_prob, entropy
    
    def update_parameters(self, loader, sigma):
        total_mse_loss = 0.0 # Initialize
        num_batches = 0      # Initialize

        for i in range(1):
            for states, actions, next_states, dist, advantage, old_log_probs, C, c in loader:
                n_batch = states.shape[0]
                advantage = advantage.double()
                f = self.Dist_func(dist, current = True) # T-1 x n_batch x n_state
                opt_states, opt_actions = self.forward(states, f, C.transpose(0, 1), c.transpose(0, 1), current = True) # x, u: T x N x Dim.
                log_probs, entropies = self.evaluate_action(opt_actions[0], actions, sigma)
        
                tau = torch.cat([states, actions], 1) # n_batch x (n_state + n_ctrl)
                nState_est = torch.bmm(self.F_hat.repeat(n_batch, 1, 1), tau.unsqueeze(-1)).squeeze(-1) + f[0] # n_batch x n_state
                mse_loss = torch.mean((nState_est - next_states)**2)
                #print("nState_est"  , nState_est)
                #print("next_states", next_states)
                print("MSE Loss: ", mse_loss.item())
                total_mse_loss += mse_loss.item()
                num_batches += 1

                #if there is already not a csv file called gnu_rl_mse_loss.csv, create it and write the mse_loss to it using cur_time 
                if not os.path.exists('gnu_rl_mse_loss.csv'):
                    with open('gnu_rl_mse_loss.csv', 'w') as f:
                        f.write('mse_loss\n')
                else:
                    with open('gnu_rl_mse_loss.csv', 'a') as f:
                        f.write(f'{mse_loss.item()}\n')


                ratio = torch.exp(log_probs.squeeze()-old_log_probs)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param) * advantage
                loss  = -torch.min(surr1, surr2).mean()
                self.optimizer.zero_grad()
                loss.backward()
                #nn.utils.clip_grad_norm_([self.F_hat, self.Bd_hat], 100)
                self.optimizer.step()
            
            self.F_hat_old = self.F_hat.detach().clone()
            self.Bd_hat_old = self.Bd_hat.detach().clone()
            print(self.F_hat)
            print(self.Bd_hat)
        # Calculate and return average RMSE for this update call
        if num_batches > 0:
            avg_mse_loss = total_mse_loss / num_batches
            avg_rmse_loss = np.sqrt(avg_mse_loss)
            return avg_rmse_loss
        return None # Return None if no batches were processed
    def Dist_func(self, d, current = False):
        if current: # d in n_batch x n_dist x T-1
            n_batch = d.shape[0]
            print("d first value: ", d[:,0])
            ft = torch.bmm(self.Bd_hat.repeat(n_batch, 1, 1), d) # n_batch x n_state x T-1
            ft = ft.transpose(1,2) # n_batch x T-1 x n_state
            ft = ft.transpose(0,1) # T-1 x n_batch x n_state
        else: # d in n_dist x T-1
            print("d first value: ", d[:,0])
            ft = torch.mm(self.Bd_hat_old, d).transpose(0, 1) # T-1 x n_state
            ft = ft.unsqueeze(1) # T-1 x 1 x n_state
        print("ft first value: ", ft[0])
        return ft

    def Cost_function(self, cur_time):
        diag = torch.zeros(args.T, self.n_state + self.n_ctrl)
        occupied = self.target["reaTSetHea_y"][cur_time:cur_time + pd.Timedelta(seconds = (args.T-1) * args.step)] == 21
        occupied = np.array(occupied)


        #print("Occupied: ", occupied)
        #print("target: ", self.target["reaTSetHea_y"][cur_time:cur_time + pd.Timedelta(seconds = (T-1) * step)])
        if len(occupied)<args.T:
            occupied = np.pad(occupied, ((0, args.T-len(occupied)), ), 'edge')
        eta_w_flag = torch.tensor([eta_occ[int(flag)] for flag in occupied]).unsqueeze(1).double() # Tx1

        diag[:, :self.n_state] = eta_w_flag
        diag[:, self.n_state:] = args.R_hat

        #print("diag values for MPC window:", diag)
        C = []
        for i in range(args.T):
            C.append(torch.diag(diag[i]))
        C = torch.stack(C).unsqueeze(1) # T x 1 x (m+n) x (m+n)
        
        #print("C", C)
        x_target = self.target[cur_time : cur_time + pd.Timedelta(seconds = (args.T-1) * args.step)] # in pd.Series
        x_target = np.array(x_target)
        #print("x target values:", x_target)
        if len(x_target)<args.T:
            x_target = np.pad(x_target, ((0, args.T-len(x_target)), (0, 0)), 'edge')
        x_target = torch.tensor(x_target)
        
        c = torch.zeros(args.T, self.n_state+self.n_ctrl) # T x (m+n)
        c[:, :self.n_state] = -eta_w_flag*x_target
        c[:, self.n_state:] = 0.1 # L1-norm now! Check
        c = c.unsqueeze(1) # T x 1 x (m+n)
        return C, c
    
def testing_loop(args):

    T=args.T
    # --- CONFIGURE these paramaters based on your choice of imitation learning hyperparameter combination---
    IL_ETA_FOR_PATH = args.IL_ETA_FOR_PATH
    IL_LR_FOR_PATH = args.IL_LR_FOR_PATH
    IL_LAMDA_FOR_PATH = args.IL_LAMDA_FOR_PATH
    IL_FHAT_EPOCH = args.IL_FHAT_EPOCH
    
    il_results_base_path = f"./results_boptest_{IL_ETA_FOR_PATH}_{IL_LR_FOR_PATH}_{IL_LAMDA_FOR_PATH}"
    DEFAULT_FHAT_PATH = os.path.join(il_results_base_path, "weights", f"F-{IL_FHAT_EPOCH}.npy")
    DEFAULT_BDHAT_PATH = os.path.join(il_results_base_path, "weights", f"Bd-{IL_FHAT_EPOCH}.npy")
    norm_params_path = "disturbance_params.pkl"
    observations_csv_path = "observations_start_day_45.csv"

    SIMULATION_YEAR_START_DATETIME = pd.Timestamp('2021-01-01 00:00:00')
    test_sim_start_time = 45*24*3600
    url = 'https://api.boptest.net'


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


    eta = args.eta
    step = args.step # step: Timestep; Unit in seconds
    T = args.T # T: Number of timesteps in the planning horizon
    tol_eps = 29 # tol_eps: Total number of episodes; Each episode is a natural day

    u_upper = 1
    u_lower = 0

    with open(norm_params_path, 'rb') as f:
        norm_params_loaded = pickle.load(f)
    disturbance_min_il = norm_params_loaded['min']
    disturbance_range_il = norm_params_loaded['range']

    obs_data = pd.read_csv(observations_csv_path, parse_dates=['timestamp'])
    obs_data.set_index('timestamp', inplace=True)

    target = obs_data[target_name]
    target_celcius = target[target_name].copy()
    target_celcius[target_name] = target_celcius[target_name] - 273.15  # Convert Kelvin to Celsius


    disturbance = obs_data[dist_name]
    disturbance['weaSta_reaWeaTDryBul_y']= disturbance['weaSta_reaWeaTDryBul_y'] - 273.15  # Convert Kelvin to Celsius
    print("disturbance unnormalized data: ", disturbance.head())
    # 8. Min-Max Normalization for disturbances (if n_dist > 0)
    if n_dist > 0 and not disturbance.empty:
        disturbance_min = disturbance_min_il
        disturbance = (disturbance - disturbance_min) / disturbance_range_il

    print(disturbance.iloc[int(45*24*3600/step):int(46*24*3600//step)])


    memory = Replay_Memory()

    #read F_hat and Bd_hat from pre-trained IL model
    F_hat_path = DEFAULT_FHAT_PATH
    Bd_hat_path = DEFAULT_BDHAT_PATH
    if os.path.exists(F_hat_path):
        F_hat = np.load(F_hat_path)
        print("F_hat value: ", F_hat)
    if os.path.exists(Bd_hat_path):
        Bd_hat = np.load(Bd_hat_path)
        print("Bd_hat value: ", Bd_hat)
    agent = PPO(memory, T, n_ctrl, n_state, target_celcius, disturbance, eta, u_upper, u_lower, F_hat =  F_hat, Bd_hat = Bd_hat)

    dir = f'online_w_rmse/online{args.run}_{args.eta}'
    if not os.path.exists(dir):
        os.mkdir(dir)

    kpi_dir = f"{dir}/kpi_all.csv"
    if os.path.exists(kpi_dir):
        "file exists, return none"
        return None


    perf = []
    multiplier = 10 # Normalize the reward for better training performance
    n_step = int(24 * 3600 / args.step)  # Number of steps in one episode (1 day)

    boptest_obs_config_for_env = {
        'reaTZon_y': (273.15, 323.15),
        'weaSta_reaWeaTDryBul_y': (250, 320),
        'weaSta_reaWeaHGloHor_y': (0, 1200),
        'reaTSetHea_y': (273.15, 310),
        'reaPHeaPum_y': (0,5000)
    }

    obs_name = list(boptest_obs_config_for_env.keys())
    keys_to_convert_to_celsius = ['reaTZon_y', 'weaSta_reaWeaTDryBul_y', 'reaTSetHea_y']

    obs_celcius = obs_data[obs_name].copy()
    for key in keys_to_convert_to_celsius:
        obs_celcius[key] = obs_celcius[key] - 273.15  # Convert Kelvin to Celsius


    env = BoptestGymEnv(
        url=url, actions=ctrl_name,
        observations=boptest_obs_config_for_env,
        max_episode_length=tol_eps * 24 * 3600,
        start_time = test_sim_start_time,
        #warmup_period=boptest_warmup_period * 24 * 3600,
        step_period=args.step,
        random_start_time=False
    )

    done = False
    obs, _ = env.reset()

    print("Environment reset complete. Starting simulation...")
    print("Initial observation:", obs)

    start_time = SIMULATION_YEAR_START_DATETIME + pd.Timedelta(seconds=test_sim_start_time)

    # Get the last date available in the obs_data
    last_date = obs_data.index[-1]
    # Compute the number of days from the start of the simulation to the last date available in the obs_data
    days_from_start = (last_date - start_time).total_seconds() / (24 * 3600)
    print(f"Number of days from the start of the simulation to the last date: {days_from_start}")



    cur_time = start_time
    print(cur_time)
    obs_dict_K = make_dict(obs_name, obs)
    obs_first_celcius = obs_dict_K.copy()
    for key in keys_to_convert_to_celsius:
        obs_first_celcius[key] = obs_first_celcius[key] - 273.15  # Convert Kelvin to Celsius

    state = torch.tensor([obs_first_celcius[name] for name in state_name]).unsqueeze(0).double() # 1 x n_state
    print("Initial state:", state)
    # Save for record
    timeStamp = [start_time]
    observations = [np.array(list(obs_first_celcius.values()))]
    actions_taken = []
    daily_rmse_records = []
    for i_episode in range(tol_eps):
        log_probs = []
        rewards = []
        real_rewards = []
        old_log_probs = []
        states = [state]
        disturbance = []
        actions = [] # Save for Parameter Updates
        CC = []
        cc = []
        sigma = args.sigma_init - 0.019*i_episode/tol_eps

        for t in range(n_step):

            dt = np.array(agent.dist[cur_time : cur_time + pd.Timedelta(seconds = (agent.T-2) * agent.step)]) # T-1 x n_dist
            dt = torch.tensor(dt).transpose(0, 1) # n_dist x T-1
            #print("Disturbance: ", dt)
            ft = agent.Dist_func(dt) # T-1 x 1 x n_state
            C, c = agent.Cost_function(cur_time)
            #print("state: ", state)
            opt_states, opt_actions = agent.forward(state, ft, C, c, current = False) # x, u: T x 1 x Dim.
            predicted_next_state = opt_states[1].squeeze(0) # 1 x n_state
            
            action, old_log_prob = agent.select_action(opt_actions[0], sigma)

            if action.item()<0:
                action = torch.zeros_like(action)
            if action.item()>1:
                action = torch.ones_like(action)

            hp_cycle = action.item()

            obs,reward,terminated,truncated,info = env.step(action)
            print("Observation from the environment: ", obs)


            obs_dict_K = make_dict(obs_name, obs)
            obs_dict_C = obs_dict_K.copy()
            for key in keys_to_convert_to_celsius:
                obs_dict_C[key] = obs_dict_C[key] - 273.15

            print("Predicted next state: ", predicted_next_state, "observed state: ", obs_dict_C['reaTZon_y'])

            cur_time = cur_time + pd.Timedelta(seconds = args.step)
            print("Current time:", cur_time)
            print(" obs_dict_C['reaTSetHea_y", obs_dict_C['reaTSetHea_y'].item())
            reward = R_func(obs_dict_C, action, eta_occ)
            
            # Per episode
            real_rewards.append(reward)
            rewards.append(reward/ multiplier)
            state = torch.tensor([obs_dict_C[name] for name in state_name]).unsqueeze(0).double()
            actions.append(action)
            old_log_probs.append(old_log_prob)
            states.append(state)
            disturbance.append(dt)
            CC.append(C.squeeze())
            cc.append(c.squeeze())
            
            # Save for record
            timeStamp.append(cur_time)
            rounded_values = np.round(np.array(list(obs_dict_C.values())), 4)
            observations.append(rounded_values)
            #print("observations shape", np.shape(np.array(observations)))
            actions_taken.append([hp_cycle, obs_dict_C['reaPHeaPum_y']])  # Save the action and heat pump power
            print("{}, Action: {:.2f}, heat pump cycle: {:.2f}, State: {:.2f}, Target: {:.2f}, Reward: {:.2f}, Power: {:.2f}".format(
                cur_time, action.item(), hp_cycle, obs_dict_C['reaTZon_y'].item(), obs_dict_C['reaTSetHea_y'], reward, obs_dict_C['reaPHeaPum_y']))

        advantages = Advantage_func(rewards, args.gamma)
        old_log_probs = torch.stack(old_log_probs).squeeze().detach().clone()
        next_states = torch.stack(states[1:]).squeeze(1)
        states = torch.stack(states[:-1]).squeeze(1)
        actions_tensor = torch.stack(actions).squeeze(1).detach().clone()
        CC = torch.stack(CC).squeeze() # n_batch x T x (m+n) x (m+n)
        cc = torch.stack(cc).squeeze() # n_batch x T x (m+n)
        disturbance = torch.stack(disturbance) # n_batch x T x n_dist
        agent.memory.append(states, actions_tensor, next_states, advantages, old_log_probs, disturbance, CC, cc)

        # if -1, do not update parameters
        if args.update_episode == -1:
            print("Pass")
            pass
        elif (agent.memory.len>= args.update_episode)&(i_episode % args.update_episode ==0):
            batch_states, batch_actions, b_next_states, batch_dist, batch_rewards, batch_old_logprobs, batch_CC, batch_cc = agent.memory.sample_batch(args.update_episode)
            batch_set = Dataset(batch_states, batch_actions, b_next_states, batch_dist, batch_rewards, batch_old_logprobs, batch_CC, batch_cc)
            batch_loader = data.DataLoader(batch_set, batch_size=48, shuffle=True, num_workers=2)
                        # Capture the returned RMSE
            current_day_rmse = agent.update_parameters(batch_loader, sigma)
            if current_day_rmse is not None:
                # Record the current date and the RMSE
                # Use cur_time.date() to get just the date part
                daily_rmse_records.append({'date': cur_time.date(), 'rmse_loss': current_day_rmse})
                print(f"Daily RMSE Loss at {cur_time.date()}: {current_day_rmse:.4f}") # Optional print
        
        perf.append([np.mean(real_rewards), np.std(real_rewards)])
        print("{}, reward: {}".format(cur_time, np.mean(real_rewards)))

        save_name = args.save_name
        obs_df = pd.DataFrame(np.array(observations), index = np.array(timeStamp), columns = obs_name)
        action_df = pd.DataFrame(np.array(actions_taken), index = np.array(timeStamp[:-1]), columns = ["oveHeaPumY_u", "reaPHeaPum_y"])
        obs_df.to_pickle(f"{dir}/perf_"+save_name+"_obs.pkl")
        action_df.to_pickle(f"{dir}/perf_"+save_name+"_actions.pkl")
        pickle.dump(np.array(perf), open(f"{dir}/perf_"+save_name+".npy", "wb"))

    kpi_dict= env.get_kpis() # <-- KPIs for the "whole period"
    kpi_df = pd.DataFrame(kpi_dict, index=[cur_time])
    kpi_df.to_csv(f"{dir}/kpi_all.csv", mode='a', header=not os.path.exists(f"{dir}/kpi_all.csv"))
    # Save daily RMSE losses
    if daily_rmse_records: # Only save if there are records
        rmse_df = pd.DataFrame(daily_rmse_records)
        rmse_file_path = os.path.join(dir, 'daily_rmse_loss.csv') # Use os.path.join for robust path
        rmse_df.to_csv(rmse_file_path, index=False)
        print(f"Daily RMSE losses saved to {rmse_file_path}")


if __name__ == "__main__":

    class Args:
        def __init__(self):
            self.gamma = 0.98  # discount factor
            self.seed = 42  # random seed
            self.lr = 5e-4  # learning rate
            self.update_episode = 1  # PPO update episode; if -1, do not update weights
            self.T = 24  # planning horizon
            self.step = 1800  # time step in simulation, unit in seconds (default: 900)
            self.save_name = 'rl'  # save name
            self.eta = 1  # hyperparameter for balancing comfort and energy
            self.R_hat = 0.000001
            self.run = 1  # run number for saving results
            self.sigma_init = 0.02 #exploration noise
            self.IL_ETA_FOR_PATH = 0.1 # chosen eta (O_occ) value for the imitation learning 
            self.IL_LR_FOR_PATH = 0.005 # chosen learning rate value for the imitation learning 
            self.IL_LAMDA_FOR_PATH = 0.1 # chosen lamda value for the imitation learning
            self.IL_FHAT_EPOCH = 49 # chosen epoch for the imitation learning

    args = Args()

    for eta in [0.1]:
        args.eta = eta
        print(f"Running PPO with eta: {args.eta}")
        for run in range(1,11):
            args.run = run
            print(f"Running PPO with run number: {run}")
            torch.manual_seed(args.run)    

            eta_occ = [0.0001, args.eta] #[0.0001, args.eta]

            testing_loop(args)
            print(f"Run {run} completed.")
            