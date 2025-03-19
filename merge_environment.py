import gymnasium
import highway_env
from matplotlib import pyplot as plt
import config_merge
import numpy as np
from highway_env.envs.merge_env import MergeEnv
from stable_baselines3 import DQN
import torch
import torch.nn as nn

def euclidian_distance(pos_1, pos_2):
    return np.linalg.norm(np.array(pos_1) - np.array(pos_2))

# Custom environment with modified reward function
class CustomMergeEnv(MergeEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
            
        #Set vehicle start speed between 5 and 15 m/s
        #Set min of 5 and max of 15 m/s 
        for vehicle in self.road.vehicles:
            vehicle.speed = np.random.randint(config_merge.initial_min_speed, config_merge.initial_max_speed)
            vehicle.MIN_SPEED = config_merge.min_speed
            vehicle.MAX_SPEED = config_merge.max_speed

    def _reward(self, action):

        reward = 0.0
        
        #Collisions
        if self.vehicle.crashed:
            reward += config_merge.w1 * -1

        #Speed
        reward += config_merge.w2 * ((self.vehicle.MAX_SPEED - np.sqrt((self.vehicle.speed - self.vehicle.MAX_SPEED)**2))/self.vehicle.MAX_SPEED)

        #Rear/Lateral        
        for vehicle in self.road.vehicles:
            if vehicle != self.vehicle:
                distance = euclidian_distance(self.vehicle.position, vehicle.position)
                #Lateral
                if vehicle.lane != self.vehicle.lane:
                    if distance < config_merge.safety_distance:
                        reward += config_merge.w3 * (-1/distance)
                #Rear
                else:
                    if distance < config_merge.safety_distance:
                        reward += config_merge.w4 * (-1/distance)

        return reward
    
    #Need to add 370 meter stopping constraint into the env
    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        if euclidian_distance(self.start_pos, self.vehicle.position) >= config_merge.max_distance:
            done = True
        return obs, reward, done, truncated, info
    
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
    
        #Initialize start position at the beginning of each episode
        self.start_pos = np.copy(self.vehicle.position)

        return obs, info

# Register the custom environment
gymnasium.register(id="custom-merge-v0", entry_point="__main__:CustomMergeEnv")

env = gymnasium.make("custom-merge-v0", render_mode="rgb_array", config={"other_vehicles_type": config_merge.other_vehicles_type,
                                                                  "observation": {"type": config_merge.observation_type, "vehicles_count": config_merge.observation_vehicles_count, "features": config_merge.observation_features},
                                                                  "action": {"type": config_merge.action_type}})

#Create model

########################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)  # Check if GPU is available
# #Uncomment when training a new model
# policy_kwargs = dict(net_arch=[64, 64], activation_fn=nn.ReLU)

# model = DQN("MlpPolicy", env, policy_kwargs=policy_kwargs, learning_rate=config_merge.learning_rate, buffer_size=config_merge.buffer_size, learning_starts=config_merge.learning_starts, batch_size=config_merge.batch_size, gamma=config_merge.gamma, train_freq=config_merge.train_frequency, exploration_fraction=config_merge.exploration_fraction, target_update_interval=config_merge.target_update_interval)
# model.learn(total_timesteps=config_merge.total_timesteps, progress_bar=True)
# model.save("DQN_Merge_Model")
########################################

model = DQN.load("DQN_Merge_Model", env=env)

obs, info = env.reset()
done = False

while not done:
    action, next_state = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()