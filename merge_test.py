import gymnasium
import highway_env
from matplotlib import pyplot as plt
import config_merge
import numpy as np
from highway_env.envs.merge_env import MergeEnv
from stable_baselines3 import DQN
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

        #Set vehicle start speed between 5 and 15 m/s
        #Set min of 5 and max of 15 m/s 
        for vehicle in self.road.vehicles:
            vehicle.speed = np.random.randint(config_merge.initial_min_speed, config_merge.initial_max_speed)
            vehicle.MIN_SPEED = config_merge.min_speed
            vehicle.MAX_SPEED = config_merge.max_speed

        return obs, info

# Register the custom environment
gymnasium.register(id="custom-merge-v0", entry_point="__main__:CustomMergeEnv")

env = gymnasium.make("custom-merge-v0", render_mode="rgb_array", config={"other_vehicles_type": config_merge.other_vehicles_type,
                                                                  "observation": {"type": config_merge.observation_type, "vehicles_count": config_merge.observation_vehicles_count, "features": config_merge.observation_features},
                                                                  "action": {"type": config_merge.action_type}})

#Create model

########################################
# #Uncomment when training a new model
policy_kwargs = dict(net_arch=[64, 64], activation_fn=nn.ReLU)

model = DQN("MlpPolicy", env, policy_kwargs=policy_kwargs, learning_rate=config_merge.learning_rate, buffer_size=config_merge.buffer_size, learning_starts=config_merge.learning_starts, batch_size=config_merge.batch_size, gamma=config_merge.gamma, train_freq=config_merge.train_frequency, exploration_fraction=config_merge.exploration_fraction, target_update_interval=config_merge.target_update_interval)
model.learn(total_timesteps=config_merge.total_timesteps, progress_bar=True)
model.save("DQN_Merge_Model")
########################################

model = DQN.load("DQN_Merge_Model", env=env)

collisions = 0
avg_speed = 0.0
avg_min_distance = 0.0
num_episodes = 10000

for episode in range(num_episodes):
    print(f"Episode: {episode}")
    obs, info = env.reset()
    done = False
    
    episode_speed_sum = 0.0
    episode_min_distance_sum = 0.0
    episode_steps = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        episode_speed_sum += env.unwrapped.vehicle.speed
        print(f"Speed: {env.unwrapped.vehicle.speed}")

        min_distance = config_merge.max_distance
        for vehicle in env.unwrapped.road.vehicles:
            if vehicle != env.unwrapped.vehicle and euclidian_distance(env.unwrapped.vehicle.position, vehicle.position) < min_distance:
                min_distance = euclidian_distance(env.unwrapped.vehicle.position, vehicle.position)
            
        episode_min_distance_sum += min_distance
        episode_steps += 1
    
    collisions += int(env.unwrapped.vehicle.crashed)
    if episode_steps > 0:
        avg_min_distance += (episode_min_distance_sum / episode_steps)

    if episode_steps > 0:
        avg_speed += (episode_speed_sum / episode_steps)

collision_rate = collisions / num_episodes
average_speed = avg_speed / num_episodes
average_min_distance = avg_min_distance / num_episodes

print(f"Collision Rate: {collision_rate*100}%")
print(f"Average Speed: {average_speed} m/s")
print(f"Average Nearest Vehicle Distance: {average_min_distance} meters")