import gymnasium
import highway_env
from matplotlib import pyplot as plt
import pprint
import config

env = gymnasium.make('merge-v0', render_mode='rgb_array', config={"other_vehicle_type": config.other_vehicles_type})

#Change environment after configuring if needed
#Example
#env.unwrapped.config["lanes_count"] = 2
#Must reset to ensure changes are made

env.reset()

for _ in range(config.time_steps):
    action = env.unwrapped.action_type.actions_indexes["IDLE"]
    obs, reward, done, truncated, info = env.step(action)
    env.render()

pprint.pprint(env.unwrapped.config)