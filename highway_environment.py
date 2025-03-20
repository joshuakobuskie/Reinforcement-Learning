import gymnasium
import highway_env
from matplotlib import pyplot as plt
import config_highway
import numpy as np
from highway_env.envs.highway_env import HighwayEnv
from stable_baselines3 import DQN
import config_tensorboard
import torch
import torch.nn as nn

def euclidian_distance(pos_1, pos_2):
    return np.linalg.norm(np.array(pos_1) - np.array(pos_2))

# Custom environment with modified reward function
class CustomHighwayEnv(HighwayEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
            
        #Set vehicle start speed between 5 and 15 m/s
        #Set min of 5 and max of 15 m/s 
        for vehicle in self.road.vehicles:
            vehicle.speed = np.random.randint(config_highway.initial_min_speed, config_highway.initial_max_speed)
            vehicle.MIN_SPEED = config_highway.min_speed
            vehicle.MAX_SPEED = config_highway.max_speed

    def _reward(self, action):

        reward = 0.0
        
        #Collisions
        if self.vehicle.crashed:
            reward += config_highway.w1 * -1

        #Speed
        reward += config_highway.w2 * ((self.vehicle.MAX_SPEED - np.sqrt((self.vehicle.speed - self.vehicle.MAX_SPEED)**2))/self.vehicle.MAX_SPEED)

        #Rear/Lateral        
        for vehicle in self.road.vehicles:
            if vehicle != self.vehicle:
                distance = euclidian_distance(self.vehicle.position, vehicle.position)
                #Lateral
                if vehicle.lane != self.vehicle.lane:
                    if distance < config_highway.safety_distance:
                        reward += config_highway.w3 * (-1/distance)
                #Rear
                else:
                    if distance < config_highway.safety_distance:
                        reward += config_highway.w4 * (-1/distance)

        return reward
    
    #Need to add 370 meter stopping constraint into the env
    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        if euclidian_distance(self.start_pos, self.vehicle.position) >= config_highway.max_distance:
            done = True
        return obs, reward, done, truncated, info
    
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
    
        #Initialize start position at the beginning of each episode
        self.start_pos = np.copy(self.vehicle.position)

        return obs, info

# Register the custom environment
gymnasium.register(id="custom-highway-v0", entry_point="__main__:CustomHighwayEnv")

env = gymnasium.make("custom-highway-v0", render_mode="rgb_array", config={"other_vehicles_type": config_highway.other_vehicles_type,
                                                                  "observation": {"type": config_highway.observation_type, "vehicles_count": config_highway.observation_vehicles_count, "features": config_highway.observation_features},
                                                                  "action": {"type": config_highway.action_type}})

#Create model

# #######################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)  # Check if GPU is available
# #Uncomment when training a new model
policy_kwargs = dict(net_arch=[64, 64], activation_fn=nn.ReLU)

model = DQN("MlpPolicy",
env, 
policy_kwargs=policy_kwargs, 
learning_rate=config_highway.learning_rate, 
buffer_size=config_highway.buffer_size, 
learning_starts=config_highway.learning_starts, 
batch_size=config_highway.batch_size, 
gamma=config_highway.gamma, 
train_freq=config_highway.train_frequency, 
exploration_fraction=config_highway.exploration_fraction, 
target_update_interval=config_highway.target_update_interval,
tensorboard_log="./DQN_Highway_Model_Eval_tensorboard/"
)
model.learn(
    total_timesteps=config_highway.total_timesteps,
    progress_bar=True,
    callback=config_tensorboard.HParamCallback())
model.save("DQN_Highway_Model_Eval")
# #######################################

model = DQN.load("DQN_Highway_Model_Eval", env=env)

obs, info = env.reset()
done = False

while not done:
    action, next_state = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()