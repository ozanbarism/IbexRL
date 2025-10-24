import sys
import os
import time
import argparse
from datetime import datetime, timedelta
import numpy as np
import torch
from torch.utils import data
from RLAgent import IbexRL
from agent.utils import Replay_Memory, Dataset, cumulative_reward, ensure_3d, process_environment, setup_logging, log_data,log_parameters, generate_daily_setpoint_schedule, check_update
import pandas as pd
# remove warnings coming from userwarning
import warnings
warnings.filterwarnings("ignore")

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

repo_path = os.path.abspath("/Users/ozanbaris/Documents/GitHub/Ibex_RL/Ibex_RL/agent")  # Adjust this path as necessary to locate your repo's root
sys.path.insert(0, repo_path)

def main():

    obs_name = ['temperature_ebtron', 'temperature (degC)',
                            'surfacesolarradiation (kW/m^2)', 'COP']
    state_name = ['temperature_ebtron']
    dist_name = ['temperature (degC)', 
        'surfacesolarradiation (kW/m^2)']
    ctrl_name = ['AC_unitout','AHU_main', 'AHU_Aux']
    target_name = ['heating_setpoint']
    
    n_state = len(state_name)
    n_ctrl = len(ctrl_name)

    step = args.step # step: Timestep; Unit in seconds
    T = args.T # T: Number of timesteps in the planning horizon
    tol_eps = 30 # tol_eps: Total number of episodes; Each episode is a natural day

    u_upper = [4.2, 10, 9.4]
    u_lower = [0.01 ,0.01 ,0.01]


    torch.manual_seed(args.seed)
    memory = Replay_Memory()
    
    param_dir = os.path.join(repo_path, "latest_parameters")
    print(f"Resolved parameter directory: {param_dir}")
    # CODE IS CHANGED TO READ PARAMETERS AS A DICTIONARY.
    parameters = {
        "Rm": np.load(os.path.join(param_dir, "Rm.npy")),
        "Rout": np.load(os.path.join(param_dir, "Rout.npy")),
        "Capacitance": np.load(os.path.join(param_dir, "Capacitance.npy")),
        "Tm": np.load(os.path.join(param_dir, "Tm.npy")),
        "Ai": np.load(os.path.join(param_dir, "Ai.npy")),
        "eta_aux": np.load(os.path.join(param_dir, "eta_aux.npy")),
        "O_hat": np.load(os.path.join(param_dir, "O_hat.npy")),
        "R_hat": np.load(os.path.join(param_dir, "R_hat.npy"))
    }
    print("Parameters are loaded")

    agent = IbexRL(memory, T, n_ctrl, n_state, u_upper, u_lower, parameters, clip_param = 0.1)
    
    # CSV log files setup
    data_filename = "data_log.csv"
    params_filename = "params_log.csv"
    setup_logging(data_filename, params_filename)


   # Save for Parameter Updates
    rewards = []
    states = []
    disturbance = []
    target_values = []
    actions = [] 
    next_states = []
    while True:
        #read the current time.
        cur_time = datetime.now().replace(minute=0, second=0, microsecond=0)

        n_batch = 1
        #disturbance_file = 'Defrost controller/oikolab-weather-output.csv'
        disturbance_file = os.path.join(repo_path, "Defrost controller/oikolab-weather-output.csv")
        indoor_temp_file = os.path.join(repo_path, "Defrost controller/indoor_temp.csv")
        setpoint_file = os.path.join(repo_path, "Defrost controller/setpoint.txt")

        # Call the function
        dt, extracted_targets, indoor_temp, setpoint = process_environment(cur_time, agent, disturbance_file, indoor_temp_file)
        
        #read the state from observations
        state = torch.tensor([indoor_temp], dtype=torch.double).unsqueeze(0)

        ft = agent.Dist_func(dt) # T-1 x 1 x n_state
        F_hat_repeated = agent.Infuse_COP(dt)
        C, c = agent.Cost_function(targets=extracted_targets, n_batch=n_batch)

        opt_states, opt_actions = agent.forward(state, F_hat_repeated, ft, C, c, n_batch, current = False) # x, u: T x 1 x Dim.

        opt_state=opt_states[1].item()
        action=opt_actions[0]

        #round assigned setpoint to the nearest 0.5
        assigned_setpoint = round(opt_state * 2) / 2
        if assigned_setpoint_setpoint < 16: 
            assigned_setpoint_setpoint = 16
        
        #save it as setpoint.txt
        with open(setpoint_file, 'w') as f:
            f.write(str(assigned_setpoint))

        #reward = R_func(obs_dict, setpoint, action, eta, cur_time) #TODO: Update the REWARD FUNCTION BEFORE DEPLOYMENT.
        reward = cumulative_reward(opt_states,extracted_targets,opt_actions)

        # Handle multi-element tensor
        action_str = ", ".join(map(str, action.squeeze().tolist()))
        print("{}, Action: [{}], Setpoint Assigned: {}, State: {}, Reward: {}".format(cur_time,
            action_str, assigned_setpoint, indoor_temp, reward))
        #wait until the top of the hour
        print("Sleeping until the top of the hour")
        #Logging data at every hour. 
        action_array = action.squeeze().tolist()
        #get the value from reward tensor
        reward_val = reward.item()
        log_data(data_filename, cur_time, indoor_temp, assigned_setpoint, reward_val, opt_state, action_array, setpoint)
        time.sleep(3600 - datetime.now().second)  
        #sleep 1 minute
        #time.sleep(60 - datetime.now().second)

        # Load indoor temperature data
        indoor_data = pd.read_csv(indoor_temp_file, parse_dates=['Time'])
        indoor_data.columns = indoor_data.columns.str.strip()
        next_state_arr = indoor_data['Value'].iloc[0] 
        next_state_temp = torch.tensor([next_state_arr], dtype=torch.double).unsqueeze(0)

        rewards.append(reward)
        actions.append(action)
        states.append(state)
        disturbance.append(dt)
        target_values.append(extracted_targets)
        next_states.append(next_state_temp)

        if check_update(cur_time, params_filename) :
            next_states = torch.stack(next_states).squeeze(1)
            states = torch.stack(states).squeeze(1)
            actions = torch.stack(actions).squeeze(1).detach().clone()
            target_values = torch.stack(target_values) 
            all_rewards = torch.tensor(rewards).unsqueeze(1)        
            disturbance = torch.stack(disturbance) # n_batch x T x n_dist
            agent.memory.append(states, actions, next_states, all_rewards, disturbance, target_values) #, CC, cc)

            print("Updating parameters at time: ", cur_time)
            batch_states, batch_actions, b_next_states, batch_dist, batch_rewards, batch_targets = agent.memory.sample_batch(args.update_episode)
     
            batch_states = batch_states.detach()
            batch_actions = batch_actions.detach()
            b_next_states = b_next_states.detach()
            batch_dist = batch_dist.detach()
            batch_rewards = batch_rewards.detach()

            batch_set = Dataset(batch_states, batch_actions, b_next_states, batch_dist, batch_rewards, batch_targets) #batch_CC, batch_cc)
            batch_loader = data.DataLoader(batch_set, batch_size=24, shuffle=False, num_workers=2)
            agent.update_parameters(batch_loader)
            parameters = {
                "Rm": agent.Rm.detach().numpy() ,
                "Rout": agent.Rout.detach().numpy() ,
                "Capacitance": agent.C.detach().numpy() ,
                "Tm": agent.Tm.detach().numpy() ,
                "Ai": agent.Ai.detach().numpy() ,
                "eta_aux": agent.eta_aux.detach().numpy() ,
                "O_hat": agent.O_hat.detach().numpy() ,
                "R_hat": agent.R_hat.detach().numpy() 
            }
            
            log_parameters(params_filename, cur_time, parameters)

            rewards = []
            states = []
            disturbance = []
            target_values = []
            actions = [] # Save for Parameter Updates
            next_states = []


if __name__ == "__main__":
    main()