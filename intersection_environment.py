import gymnasium
import highway_env
from matplotlib import pyplot as plt
import config_intersection
import numpy as np
from highway_env.envs.intersection_env import IntersectionEnv
from stable_baselines3 import DQN
import torch
import torch.nn as nn

def euclidian_distance(pos_1, pos_2):
    return np.linalg.norm(np.array(pos_1) - np.array(pos_2))

# Custom environment with modified reward function
class CustomIntersectionEnv(IntersectionEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
            
        #Set vehicle start speed between 5 and 15 m/s
        #Set min of 5 and max of 15 m/s 
        for vehicle in self.road.vehicles:
            vehicle.speed = np.random.randint(config_intersection.initial_min_speed, config_intersection.initial_max_speed)
            vehicle.MIN_SPEED = config_intersection.min_speed
            vehicle.MAX_SPEED = config_intersection.max_speed

    def _reward(self, action):

        reward = 0.0
        
        #Collisions
        if self.vehicle.crashed:
            reward += config_intersection.w1 * -1

        #Speed
        reward += config_intersection.w2 * ((self.vehicle.MAX_SPEED - np.sqrt((self.vehicle.speed - self.vehicle.MAX_SPEED)**2))/self.vehicle.MAX_SPEED)

        #Rear/Lateral        
        for vehicle in self.road.vehicles:
            if vehicle != self.vehicle:
                distance = euclidian_distance(self.vehicle.position, vehicle.position)
                #Lateral
                if vehicle.lane != self.vehicle.lane:
                    if distance < config_intersection.safety_distance:
                        reward += config_intersection.w3 * (-1/distance)
                #Rear
                else:
                    if distance < config_intersection.safety_distance:
                        reward += config_intersection.w4 * (-1/distance)

        return reward
    
    #Need to add 370 meter stopping constraint into the env
    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        if euclidian_distance(self.start_pos, self.vehicle.position) >= config_intersection.max_distance:
            done = True
        return obs, reward, done, truncated, info
    
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
    
        #Initialize start position at the beginning of each episode
        self.start_pos = np.copy(self.vehicle.position)

        return obs, info

# Register the custom environment
gymnasium.register(id="custom-intersection-v0", entry_point="__main__:CustomIntersectionEnv")

env = gymnasium.make("custom-intersection-v0", 
    render_mode="rgb_array", 
    config={"other_vehicles_type": config_intersection.other_vehicles_type,
    "observation": {"type": config_intersection.observation_type,
    "vehicles_count": config_intersection.observation_vehicles_count,
    "features": config_intersection.observation_features},
    "action": {"type": config_intersection.action_type}})

#Create model

# #######################################
# #Uncomment when training a new model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)  # Check if GPU is available
# policy_kwargs = dict(net_arch=[64, 64], activation_fn=nn.ReLU)
# model = DQN(
#     "MlpPolicy",
#     env,
#     policy_kwargs=policy_kwargs,
#     learning_rate=config_intersection.learning_rate,
#     tensorboard_log="./DQN_Intersection_Model_Adjusted_tensorboard/",
#     buffer_size=config_intersection.buffer_size,
#     learning_starts=config_intersection.learning_starts,
#     batch_size=config_intersection.batch_size,
#     gamma=config_intersection.gamma,
#     train_freq=config_intersection.train_frequency,
#     exploration_fraction=config_intersection.exploration_fraction,
#     target_update_interval=config_intersection.target_update_interval)

# model.learn(total_timesteps=config_intersection.total_timesteps, progress_bar=True)
# model.save("DQN_Intersection_Model_Adjusted")
######################################

model = DQN.load("DQN_Intersection_Model_Adjusted", env=env)

obs, info = env.reset()
done = False

while not done:
    action, next_state = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()