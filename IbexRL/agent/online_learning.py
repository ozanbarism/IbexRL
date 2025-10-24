import pandas as pd
import warnings
import os
import sys
import numpy as np
import time
import argparse
from datetime import datetime, timedelta
import torch
from torch.utils import data
import pickle

warnings.filterwarnings("ignore")
from IbexAgent import IbexRL
from utils import make_dict, Replay_Memory, Dataset, R_func
from boptestGymEnv import BoptestGymEnv
from diff_mpc import mpc
from diff_mpc.mpc import QuadCost, LinDx


def imit_dir_gen(args):
    IL_ETA_FOR_PATH = args.IL_ETA_FOR_PATH
    IL_LR_FOR_PATH = args.IL_LR_FOR_PATH
    IL_LAMDA_FOR_PATH = args.IL_LAMDA_FOR_PATH
    IL_FHAT_EPOCH = args.IL_FHAT_EPOCH

    if args.no_constraint:
        imit_dir = f'RESULTS/IMITATION/lr{IL_LR_FOR_PATH}_eta{IL_ETA_FOR_PATH}_lamda{IL_LAMDA_FOR_PATH}_noconstraint'
    else:
        imit_dir = f'RESULTS/IMITATION/lr{IL_LR_FOR_PATH}_eta{IL_ETA_FOR_PATH}_lamda{IL_LAMDA_FOR_PATH}_constraint'

    if args.predecided_init:
        imit_dir = f"{imit_dir}_initdecided"
    else:
        imit_dir = f"{imit_dir}_init{args.init}"

    return imit_dir

def online_dir_gen(args, imit_dir):
    if args.no_constraint:
        dir = f'./RESULTS/{args.folder_name}/online{args.run}_lr{args.state_lr}_imit{imit_dir}'
    else:
        dir = f'./RESULTS/{args.folder_name}/online{args.run}_lr{args.state_lr}_imit{imit_dir}'

    if args.read_O:
        dir = f"{dir}_readO"
    else:
        dir = f"{dir}_R{args.R_hat}"

    if args.read_R:
        dir = f"{dir}_readR"
    else:
        dir = f"{dir}_O{args.eta}"

    if args.explore:
        print("Exploration mode enabled.")
        dir = f"{dir}_explore"

    if os.path.exists(f"{dir}/kpi_all.csv"):
        exist = True
        print(f"Directory {dir} already exists. skipping this test case.")
    else:
        exist = False

    return dir, exist


def test_loop(args):


    n_step = int(24 * 3600 / args.step)  # Number of steps in one episode (1 day)
    eta_occ = [args.O_unocc, args.O_hat]  # Coefficients for occupied and unoccupied states
    path_for_observations= f"{args.gnu_rl_path}/observations.csv"
    SIMULATION_YEAR_START_DATETIME = pd.Timestamp('2021-01-01 00:00:00')
    test_sim_start_time = 45*24*3600
    url = 'https://api.boptest.net'
    # Modify here: Based on your specific control problem and available BOPTEST data
    state_name = ['reaTZon_y'] # Zone operative temperature

    # Disturbances available from your configured BOPTEST observations
    dist_name = [
        'weaSta_reaWeaTDryBul_y',   # Maps to "Outdoor Temp."
        'weaSta_reaWeaHGloHor_y',    # Maps to a "Solar Rad." component
        'reaCOP_y'
    ]

    # Control action - this is what the agent learns. It's your expert action from BOPTEST.
    ctrl_name = ['oveHeaPumY_u'] # Directly use the heat pump action

    # Target for the state variable (e.g., the setpoint for indoor temperature)
    target_name = ['reaTSetHea_y'] # Using heating setpoint as the primary target.
                                # You could also use 'reaTSetCoo_y' or create a
                                # combined "effective setpoint" column during data preprocessing.

    n_state = len(state_name) # Number of state variables
    n_ctrl = len(ctrl_name) # Number of control actions

    T = args.T # T: Number of timesteps in the planning horizon
    tol_eps = 29 # tol_eps: Total number of episodes; Each episode is a natural day

    u_upper = 1 # Upper bound for the control action
    u_lower = 0 # Lower bound for the control action


    obs_data = pd.read_csv(path_for_observations, parse_dates=['timestamp'])
    obs_data.set_index('timestamp', inplace=True)

    target = obs_data[target_name]
    target_celcius = target[target_name].copy()
    target_celcius[target_name] = target_celcius[target_name] - 273.15  # Convert Kelvin to Celsius


    disturbance = obs_data[dist_name]


    memory = Replay_Memory()


    imit_dir = imit_dir_gen(args)
    param_dir = os.path.join(imit_dir, 'parameters')

    # CODE IS CHANGED TO READ PARAMETERS AS A DICTIONARY.
    epoch = args.IL_FHAT_EPOCH
    parameters = {
        "Rm": np.load(os.path.join(param_dir, f"Rm-{epoch}.npy")),
        "Rout": np.load(os.path.join(param_dir, f"Rout-{epoch}.npy")),
        "Capacitance": np.load(os.path.join(param_dir, f"C-{epoch}.npy")),
        "Tm": np.load(os.path.join(param_dir, f"Tm-{epoch}.npy")),
        "Ai": np.load(os.path.join(param_dir, f"Ai-{epoch}.npy")),
        "P_max": np.load(os.path.join(param_dir, f"P_max-{epoch}.npy")),
    }
    if args.read_R:
        parameters["R_hat"] = np.load(os.path.join(param_dir, f"R-{epoch}.npy"))
    else:
         parameters["R_hat"] = np.array([args.R_hat]),#np.load(os.path.join(param_dir, f"R-{epoch}.npy")),# #


    if args.read_O:
        parameters["O_hat"] = np.load(os.path.join(param_dir, f"O-{epoch}.npy"))
    else:
        parameters["O_hat"] = np.array([args.O_hat])
    
    #remove the IMITATION/ part form imit_dir
    imit_dir = imit_dir.replace('IMITATION/', '')

    print("Parameters are loaded: ", parameters)


    dir, exist = online_dir_gen(args, imit_dir)

    if exist:
        print(f"Skipping test case for {dir} as it already exists.")
        return
    else:
        os.makedirs(dir, exist_ok=True)

    print("Directory for saving results:", dir)
    perf = []

    boptest_obs_config_for_env = {
        'reaTZon_y': (273.15, 323.15),
        'weaSta_reaWeaTDryBul_y': (250, 320),
        'weaSta_reaWeaHGloHor_y': (0, 1200),
        'reaTSetHea_y': (273.15, 310),
        'reaPHeaPum_y':(0,5000), 
    }

    obs_name = list(boptest_obs_config_for_env.keys())
    keys_to_convert_to_celsius = ['reaTZon_y', 'weaSta_reaWeaTDryBul_y', 'reaTSetHea_y']

    obs_celcius = obs_data[obs_name].copy()
    for key in keys_to_convert_to_celsius:
        obs_celcius[key] = obs_celcius[key] - 273.15  # Convert Kelvin to Celsius

    disturbance_celcius = obs_data[dist_name].copy()
    disturbance_celcius['weaSta_reaWeaTDryBul_y'] = disturbance_celcius['weaSta_reaWeaTDryBul_y'] - 273.15  # Convert Kelvin to Celsius
    disturbance_celcius['weaSta_reaWeaHGloHor_y'] = disturbance_celcius['weaSta_reaWeaHGloHor_y'] / 1000  # Convert W/m^2 to kW/m^2

    env = BoptestGymEnv(
        url=url, actions=ctrl_name,
        observations=boptest_obs_config_for_env,
        max_episode_length=tol_eps * 24 * 3600,
        start_time = test_sim_start_time,
        step_period=args.step,
        random_start_time=False
    )

    agent = IbexRL(args, memory, T, n_ctrl, n_state, u_upper, u_lower, parameters, disturbance_celcius, target_celcius, clip_param = 0.1)
    
    done = False
    obs, _ = env.reset()

    print("Environment reset complete. Starting simulation...")
    print("Initial observation:", obs)

    #start_time is simulation start datetime plus 45*24*3600 seconds
    start_time = SIMULATION_YEAR_START_DATETIME + pd.Timedelta(seconds = test_sim_start_time)
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
    daily_rmse_records = [] # New list to store date and RMSE
    for i_episode in range(tol_eps):
        rewards = []
        real_rewards = []
        actions=[]
        states = [state]
        disturbance = []
        target_values = []

        sigma = args.sigma_init - 0.019*i_episode/tol_eps
        for t in range(n_step):

            n_batch = 1
            # Ensure the index of agent.dist is a DatetimeIndex
            if not isinstance(agent.dist.index, pd.DatetimeIndex):
                agent.dist.index = pd.to_datetime(agent.dist.index)
            if not isinstance(agent.target.index, pd.DatetimeIndex):
                agent.target.index = pd.to_datetime(agent.target.index)
            #get disturbance values into the future. 
            raw_dt = agent.dist[cur_time : cur_time + pd.Timedelta(seconds=(agent.T - 2) * agent.step)]
            #print("shape of raw_dt before processing:", np.array(raw_dt).shape)
            # Convert raw data to NumPy and adjust orientation if needed
            dt = np.array(raw_dt)
            if dt.shape[0] > dt.shape[1]:  # Check if orientation is incorrect
                dt = dt.T  # Transpose to [n_dist, T-1]
            # Convert to PyTorch tensor
            dt = torch.tensor(dt, dtype=torch.double)
            # Extract values for the specified range
            extracted_targets = agent.target[cur_time : cur_time + pd.Timedelta(seconds=(agent.T -1) * agent.step)].values

            if extracted_targets.shape[0] > extracted_targets.shape[1]:
                extracted_targets = extracted_targets.T  # Transpose to [n_dist, T-1]

            extracted_targets = torch.tensor(extracted_targets, dtype=torch.double)

            ft = agent.Dist_func(dt) # T-1 x 1 x n_state
            F_hat_repeated = agent.Infuse_COP(dt)

            C, c = agent.Cost_function(targets=extracted_targets, n_batch=n_batch)

            opt_states, opt_actions = agent.forward(state, F_hat_repeated, ft, C, c, n_batch, current = False) # x, u: T x 1 x Dim.

            print("Optimal action:", opt_actions[0])
            
            if args.explore:
                action, old_log_prob = agent.select_action(opt_actions[0], sigma)

                if action.item()<0:
                    action = torch.zeros_like(action)
                if action.item()>1:
                    action = torch.ones_like(action)
            else:
                action=opt_actions[0]

            print("Action selected:", action)
            hp_cycle = action.item()

            #Take a step in the environment
            obs,reward,terminated,truncated,info = env.step(action)
            # observations come in Kelvin, convert to Celsius for better interpretability
            obs_dict_K = make_dict(obs_name, obs)
            obs_dict_C = obs_dict_K.copy()
            for key in keys_to_convert_to_celsius:
                obs_dict_C[key] = obs_dict_C[key] - 273.15

            cur_time = cur_time + pd.Timedelta(seconds = args.step)
            print("cur_time:", cur_time, "state :", obs_dict_C['reaTZon_y'], "action:", hp_cycle, "reward:", reward, "Power:", obs_dict_C['reaPHeaPum_y'])
            # When cost is calibrated, we are not using R_func (quadratic reward) but rather using the non-quadratic reward. However, here we keep R_func to report the reward used by the differentiable MPC. 
            # This reward is not used for updates so it is only for reporting purposes.
            reward = R_func(obs_dict_C, action, eta_occ, args.s_t) 

            # Per episode
            real_rewards.append(reward)
            rewards.append(reward)
            state = torch.tensor([obs_dict_C[name] for name in state_name]).unsqueeze(0).double()
            actions.append(action.squeeze(0).detach().clone()) 
            states.append(state)
            disturbance.append(dt)
            target_values.append(extracted_targets)

            # Save for record
            timeStamp.append(cur_time)
            rounded_values = np.round(np.array(list(obs_dict_C.values())), 4)
            observations.append(rounded_values)
            # Flatten action and append it to the list
            actions_taken.append([hp_cycle, obs_dict_C['reaPHeaPum_y']])


        # Append to perf
        perf.append([np.mean(real_rewards), np.std(real_rewards)])
        next_states = torch.stack(states[1:]).squeeze(1)
        states = torch.stack(states[:-1]).squeeze(1)
        actions_tensor = torch.stack(actions).detach().clone()
        target_values = torch.stack(target_values) 
        
        disturbance = torch.stack(disturbance) # n_batch x T x n_dist
        agent.memory.append(states, actions_tensor, next_states, rewards, disturbance, target_values) #, CC, cc)

        
        if args.update_episode == -1:
            print("Pass")
            pass
        elif (agent.memory.len>= args.update_episode)&(i_episode % args.update_episode ==0):

            batch_states, batch_actions, b_next_states, batch_dist, batch_rewards, batch_targets = agent.memory.sample_batch(args.update_episode)
            
            batch_set = Dataset(batch_states, batch_actions, b_next_states, batch_dist, batch_rewards, batch_targets) #batch_CC, batch_cc)
            batch_loader = data.DataLoader(batch_set, batch_size=24, shuffle=False, num_workers=0)


            # Capture the returned RMSE
            current_day_rmse = agent.update_parameters(batch_loader)
            if current_day_rmse is not None:
                # Record the current date and the RMSE
                daily_rmse_records.append({'date': cur_time.date(), 'rmse_loss': current_day_rmse})
                print(f"Daily RMSE Loss at {cur_time.date()}: {current_day_rmse:.4f}") # Optional print
        
    # After all episodes are done, save rewards to file
    save_name = args.save_name
    # Save additional metrics and observations
    obs_df = pd.DataFrame(np.array(observations), index=np.array(timeStamp), columns=obs_name)
    action_df = pd.DataFrame(np.array(actions_taken), index=np.array(timeStamp[:-1]), columns=['oveHeaPumY_u', 'reaPHeaPum_y'])
    obs_df.to_pickle(f"{dir}/perf_" + save_name + "_obs.pkl")
    action_df.to_pickle(f"{dir}/perf_" + save_name + "_actions.pkl")
    pickle.dump(np.array(perf), open(f"{dir}/perf_" + save_name + ".npy", "wb"))

    kpi_dict= env.get_kpis() # <-- KPIs for the "whole period"
    kpi_df = pd.DataFrame(kpi_dict, index=[cur_time])
    kpi_df.to_csv(f"{dir}/kpi_all.csv", mode='a', header=not os.path.exists(f"{dir}/kpi_all.csv"))
        # Save daily RMSE losses
    if daily_rmse_records: # Only save if there are records
        rmse_df = pd.DataFrame(daily_rmse_records)
        rmse_file_path = os.path.join(dir, 'daily_rmse_loss.csv')
        rmse_df.to_csv(rmse_file_path, index=False)
        print(f"Daily RMSE losses saved to {rmse_file_path}")



