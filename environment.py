import gymnasium
import highway_env
from matplotlib import pyplot as plt
import pprint
import config
import numpy as np

env = gymnasium.make('merge-v0', render_mode='rgb_array', config={"other_vehicles_type": config.other_vehicles_type, "vehicles_count": config.vehicles_count})

#Change environment after configuring if needed
#Example
#env.unwrapped.config["lanes_count"] = 2
#Must reset to ensure changes are made

#Set vehicle start speed between 5 and 15 m/s
for vehicle in env.unwrapped.road.vehicles:
    vehicle.speed = np.random.randint(config.initial_min_speed, config.initial_max_speed)
    print("Vehicle: {}, Initial Speed: {}".format(vehicle, vehicle.speed))

env.reset()

iters = 0
done = False
#Changed to stop on crash
while not done and iters < config.time_steps:
    action = env.unwrapped.action_type.actions_indexes["IDLE"]
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    iters += 1

pprint.pprint(env.unwrapped.config)