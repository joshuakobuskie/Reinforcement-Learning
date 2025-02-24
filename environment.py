import gymnasium
import highway_env
from matplotlib import pyplot as plt
import pprint
import config

env = gymnasium.make('merge-v0', render_mode='rgb_array')
env.reset()

for _ in range(config.time_steps):
    action = env.unwrapped.action_type.actions_indexes["IDLE"]
    obs, reward, done, truncated, info = env.step(action)
    env.render()

pprint.pprint(env.unwrapped.config)