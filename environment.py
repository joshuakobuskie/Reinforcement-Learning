import gymnasium
import highway_env
from matplotlib import pyplot as plt
import pprint
import config
import numpy as np

def euclidian_distance(pos_1, pos_2):
    return np.linalg.norm(np.array(pos_2) - np.array(pos_1))

env = gymnasium.make('merge-v0', render_mode='rgb_array', config={"other_vehicles_type": config.other_vehicles_type, "observation": {"type": config.observation_type, "vehicles_count": config.observation_vehicles_count, "features": config.observation_features}})
 
#Change environment after configuring if needed
#Example
#env.unwrapped.config["lanes_count"] = 2
#Must reset to ensure changes are made

#pprint.pprint(env.unwrapped)

#Set vehicle start speed between 5 and 15 m/s
for vehicle in env.unwrapped.road.vehicles:
    vehicle.speed = np.random.randint(config.initial_min_speed, config.initial_max_speed)
    print("Vehicle: {}, Initial Speed: {}".format(vehicle, vehicle.speed))

env.reset()

iters = 0
done = False
#Changed to stop on crash
while not done and iters < config.max_iters:
    action = env.unwrapped.action_type.actions_indexes["IDLE"]
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    #Confirmed working as intended: 3 objects with only x,y,vx,vy
    #print(obs)
    iters += 1

pprint.pprint(env.unwrapped.config)