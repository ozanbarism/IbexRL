'''
Module to run the baseline controller of a model using the `BoptestGymEnv`
interface. This example does not learn any policy, it is rather used
to test the environment.
The BOPTEST bestest_hydrinic_heat_pump case needs to be deployed.

'''
import numpy as np
import requests
import random
import sys
import os

#hardcode the system path to the boptestGymEnv
sys.path.append('/Users/ozanbaris/Documents/GitHub/IbexRLExperiments/Gnu-RL/project1-boptest-gym')


from boptestGymEnv import BoptestGymEnv, NormalizedObservationWrapper, NormalizedActionWrapper
from examples.test_and_plot import test_agent
import pandas as pd

# Seed for random starting times of episodes
random.seed(123456)

start_time_test    = 45*24*3600
episode_length_test = 30*24*3600
warmup_period_test  = 0
step_period_seconds = 1800


# Assume a default URL if not provided by the environment or specific setup
# This might need to be configured based on your BOPTEST setup.
# For example, if BOPTEST is running locally:
url = 'https://api.boptest.net'

def run_reward_default(plot=False):
    '''Run example with default reward function.

    Parameters
    ----------
    plot : bool, optional
        True to plot timeseries results.
        Default is False.

    Returns
    -------
    observations : list
        Observations obtained in simulation
    actions : list
        Actions applied in simulation
    rewards : list
        Rewards obtained in simulation

    '''

    observations, actions, rewards = run(envClass=BoptestGymEnv, plot=plot)

    return observations, actions, rewards

def run_reward_custom(plot=False):
    '''Run example with customized reward function.

    Parameters
    ----------
    plot : bool, optional
        True to plot timeseries results.
        Default is False.

    Returns
    -------
    observations : list
        Observations obtained in simulation
    actions : list
        Actions applied in simulation
    rewards : list
        Rewards obtained in simulation

    '''

    # Define a parent class as a wrapper to override the reward function
    class BoptestGymEnvCustom(BoptestGymEnv):

        def get_reward(self):
            '''Custom reward function that penalizes less the discomfort
            and thus more the operational cost.

            '''

            # Define relative weight for discomfort
            w = 0.1

            # Compute BOPTEST core kpis
            kpis = requests.get('{0}/kpi/{1}'.format(self.url, self.testid)).json()['payload']

            # Calculate objective integrand function at this point
            objective_integrand = kpis['cost_tot']*12.*16. + w*kpis['tdis_tot']

            # Compute reward
            reward = -(objective_integrand - self.objective_integrand)

            self.objective_integrand = objective_integrand

            return reward

    observations, actions, rewards = run(envClass=BoptestGymEnvCustom, plot=plot)

    return observations, actions, rewards

def run_reward_clipping(plot=False):
    '''Run example with clipped reward function.

    Parameters
    ----------
    plot : bool, optional
        True to plot timeseries results.
        Default is False.

    Returns
    -------
    observations : list
        Observations obtained in simulation
    actions : list
        Actions applied in simulation
    rewards : list
        Rewards obtained in simulation

    '''

    # Define a parent class as a wrapper to override the reward function
    class BoptestGymEnvClipping(BoptestGymEnv):

        def get_reward(self):
            '''Clipped reward function that has the value either -1 when
            there is any cost/discomfort, or 0 where there is not cost
            nor discomfort. This would be the simplest reward to learn for
            an agent.

            '''

            # Compute BOPTEST core kpis
            kpis = requests.get('{0}/kpi/{1}'.format(self.url, self.testid)).json()['payload']

            # Calculate objective integrand function at this point
            objective_integrand = kpis['cost_tot']*12.*16. + kpis['tdis_tot']

            # Compute reward
            reward = -(objective_integrand - self.objective_integrand)

            # Filter to be either -1 or 0
            reward = np.sign(reward)

            self.objective_integrand = objective_integrand

            return reward

    observations, actions, rewards = run(envClass=BoptestGymEnvClipping, plot=plot)

    return observations, actions, rewards

def run_normalized_observation_wrapper(plot=False):
    '''Run example with normalized observation wrapper.

    Parameters
    ----------
    plot : bool, optional
        True to plot timeseries results.
        Default is False.

    Returns
    -------
    observations : list
        Observations obtained in simulation
    actions : list
        Actions applied in simulation
    rewards : list
        Rewards obtained in simulation

    '''

    observations, actions, rewards = run(envClass=BoptestGymEnv,
                                     wrapper=NormalizedObservationWrapper,
                                     plot=plot)

    return observations, actions, rewards

def run_normalized_action_wrapper(plot=False):
    '''Run example with normalized action wrapper.

    Parameters
    ----------
    plot : bool, optional
        True to plot timeseries results.
        Default is False.

    Returns
    -------
    observations : list
        Observations obtained in simulation
    actions : list
        Actions applied in simulation
    rewards : list
        Rewards obtained in simulation

    '''

    observations, actions, rewards = run(envClass=BoptestGymEnv,
                                     wrapper=NormalizedActionWrapper,
                                     plot=plot)

    return observations, actions, rewards

def run_highly_dynamic_price(plot=False):
    '''Run example when setting the highly dynamic price scenario of BOPTEST.

    Parameters
    ----------
    plot : bool, optional
        True to plot timeseries results.
        Default is False.

    Returns
    -------
    observations : list
        Observations obtained in simulation
    actions : list
        Actions applied in simulation
    rewards : list
        Rewards obtained in simulation

    '''

    observations, actions, rewards = run(envClass=BoptestGymEnv,
                                     scenario={'electricity_price':'highly_dynamic'},
                                     plot=plot)

    return observations, actions, rewards

def run(envClass, wrapper=None, scenario={'electricity_price':'constant'},
        plot=False):

    boptest_observations_config = {
        'reaPHeaPum_y': (0, 10000),              # Heat pump electrical power (W)
        'reaCOP_y': (0, 10),                     # Heat pump COP (1)
        'reaTSetCoo_y': (280, 310),         # Zone operative temperature setpoint for cooling (K)
        'reaTSetHea_y': (280, 310),         # Zone operative temperature setpoint for heating (K)
        'reaTZon_y': (280, 310),           # Zone operative temperature (K)
        'weaSta_reaWeaHGloHor_y': (0, 1200),     # Global horizontal solar irradiation measurement (W/m2)
        'weaSta_reaWeaTDryBul_y': (250, 320),    # Outside drybulb temperature measurement (K)
        'weaSta_reaWeaTWetBul_y': (250, 320)     # Wet bulb temperature measurement (K)

    }


    # Use the first 3 days of February for testing with 3 days for initialization
    env = envClass(url                = url,
                   actions            = ['oveHeaPumY_u'],
                   observations       = boptest_observations_config,
                   random_start_time  = False,
                   start_time         = start_time_test ,
                   max_episode_length = episode_length_test,
                   warmup_period      = warmup_period_test,
                   scenario           = scenario,
                   step_period        = step_period_seconds)

    # Define an empty action list to don't overwrite any input
    env.actions = []

    # Add wrapper if any
    if wrapper is not None:
        env = wrapper(env)

    model = BaselineModel()
    # Perform test
    observations, actions, rewards, _ = test_agent(env, model,
                                     start_time=start_time_test,
                                     episode_length=episode_length_test,
                                     warmup_period=warmup_period_test,
                                     plot=plot)


    # stop the test
    env.stop()
    print("Test stopped.")

    # ---- MODIFICATION START ----

    # Save the data

    # Map the array of observations to a DataFrame using the keys from the dictionary

    observation_keys = list(boptest_observations_config.keys())

    observations_df = pd.DataFrame(observations, columns=observation_keys)



    print("Observations DataFrame created with columns:", observations_df.columns)

    # Generate timestamps based on the simulation start time and step period

    timestamps = pd.date_range(

    start=pd.Timestamp('2021-01-01'),

    periods=len(observations),

    freq=f'{step_period_seconds}S'

    )

    observations_df['timestamp'] = timestamps

    print("Timestamps added to DataFrame")

    #make the timestamp the first column

    observations_df = observations_df[['timestamp'] + observation_keys]

    # Save the DataFrame to a CSV file

    observations_df.to_csv('observations_test.csv', index=False)


    print("Observations saved to observations.csv with timestamps")

    np.save('rewards_test.npy', np.array(rewards, dtype=object))

    print("Data saved: observations.npy, actions.npy, rewards.npy")

    return observations, actions, rewards

class BaselineModel(object):
    '''Dummy class for baseline model. It simply returns empty list when
    calling `predict` method.

    '''
    def __init__(self):
        pass
    def predict(self, obs, deterministic=True):
        return [], obs

class SampleModel(object):
    '''Dummy class that generates random actions. It therefore does not
    simulate the baseline controller, but is still maintained here because
    also serves as a simple case to test features.

    '''
    def __init__(self):
        # The SampleModel needs an action_space attribute to call sample()
        # We'll mock a simple one if it's not provided by an env
        # This might need adjustment based on the actual action space structure
        from gymnasium.spaces import Box # Assuming gymnasium is used by BoptestGymEnv
        self.action_space = Box(low=0, high=1, shape=(1,), dtype=np.float32)
    def predict(self,obs, deterministic=True):
        return self.action_space.sample(), obs

if __name__ == "__main__":
    # It's good practice to ensure the BOPTEST server is running and accessible.
    # You might want to add a check for the server availability here.
    try:
        # Check if BOPTEST server is available
        requests.get(url + '/status', timeout=5) # Add timeout
        print(f"BOPTEST server is accessible at {url}")

        # Example: Run with custom reward and plot, data will be saved automatically
        print("Running simulation with custom reward function...")
        obs_data, act_data, rew_data = run_reward_custom(plot=True) # Changed variable name for clarity
        print(f"Simulation complete. {len(obs_data)} observations, {len(act_data)} actions, {len(rew_data)} rewards collected.")

        # You can now load the data if needed:
        # loaded_observations = np.load('observations.npy', allow_pickle=True)
        # loaded_actions = np.load('actions.npy', allow_pickle=True)
        # loaded_rewards = np.load('rewards.npy', allow_pickle=True)
        # print("Successfully loaded saved data for verification (optional).")

    except requests.exceptions.ConnectionError:
        print(f"Error: BOPTEST server not accessible at {url}.")
        print("Please ensure the BOPTEST environment is running.")
    except FileNotFoundError:
        print("Error: 'examples.test_and_plot' or 'boptestGymEnv' not found.")
        print("Ensure that these modules are in your Python path and BOPTEST is installed correctly.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")