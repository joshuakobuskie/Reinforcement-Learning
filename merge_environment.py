import gymnasium
import highway_env
import config_merge
import numpy as np
import random
import torch
import torch.nn as nn
from stable_baselines3 import DQN
from matplotlib import pyplot as plt
from highway_env.envs.merge_env import MergeEnv
import config_tensorboard
from tqdm import tqdm
import curriculum_scheduler
from stable_baselines3.common.callbacks import BaseCallback

def euclidian_distance(pos_1, pos_2):
    return np.linalg.norm(np.array(pos_1) - np.array(pos_2))

class CustomMergeEnv(MergeEnv):
    def __init__(self, *args, **kwargs):
        self.vehicles_count = config_merge.vehicles_count
        self.initial_min_speed = config_merge.initial_min_speed
        self.initial_max_speed = config_merge.initial_max_speed
        self.min_speed = config_merge.min_speed
        self.max_speed = config_merge.max_speed
        self.curriculum_stage = 0
        super().__init__(*args, **kwargs)

        for vehicle in self.road.vehicles:
            vehicle.speed = np.random.randint(self.initial_min_speed, self.initial_max_speed)
            vehicle.MIN_SPEED = self.min_speed
            vehicle.MAX_SPEED = self.max_speed

    def _reward(self, action):
        reward = 0.0
        if self.vehicle.crashed:
            reward += config_merge.w1 * -1
        reward += config_merge.w2 * ((self.vehicle.MAX_SPEED - np.sqrt((self.vehicle.speed - self.vehicle.MAX_SPEED) ** 2)) / self.vehicle.MAX_SPEED)
        for vehicle in self.road.vehicles:
            if vehicle != self.vehicle:
                distance = euclidian_distance(self.vehicle.position, vehicle.position)
                if vehicle.lane != self.vehicle.lane:
                    if distance < config_merge.safety_distance:
                        reward += config_merge.w4 * (-1 / distance)
                else:
                    if distance < config_merge.safety_distance:
                        reward += config_merge.w3 * (-1 / distance)
        return reward

    def set_config(self, new_config):
        for key, val in new_config.items():
            if hasattr(self, key):
                setattr(self, key, val)
        if "stage" in new_config:
            self.curriculum_stage = new_config["stage"]

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        if euclidian_distance(self.start_pos, self.vehicle.position) >= config_merge.max_distance:
            done = True
        return obs, reward, done, truncated, info

    def is_far_enough(new_pos, vehicles, min_dist=10.0):
        for v in vehicles:
            if hasattr(v, "position") and np.linalg.norm(np.array(v.position) - np.array(new_pos)) < min_dist:
                return False
        return True

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.start_pos = np.copy(self.vehicle.position)
        self.road.vehicles = [self.vehicle]
        lane_edges = []
        for from_node in self.road.network.graph:
            for to_node in self.road.network.graph[from_node]:
                lane_edges.append((from_node, to_node))

        for i in range(self.vehicles_count - 1):
            speed_clip = np.clip(
                np.random.uniform(self.initial_min_speed, self.initial_max_speed),
                self.min_speed,
                self.max_speed
            )
            from_node, to_node = random.choice(lane_edges)
            lane_count = len(self.road.network.graph[from_node][to_node])

            if self.curriculum_stage == 0:
                lane_index = 0
                x_offset = i * 30
            elif self.curriculum_stage == 1:
                lane_index = i % lane_count
                x_offset = i * 20 + np.random.uniform(-2, 2)
            else:
                lane_index = random.randint(0, lane_count - 1)
                x_offset = i * 15 + np.random.uniform(-5, 5)

            lane = self.road.network.get_lane((from_node, to_node, lane_index))
            position = lane.position(self.vehicle.position[0] + x_offset, 0)

            if not self.is_far_enough(position, self.road.vehicles):
                continue

            new_vehicle = highway_env.vehicle.behavior.IDMVehicle(
                self.road, position=position, speed=speed_clip
            )
            new_vehicle.MIN_SPEED = self.min_speed
            new_vehicle.MAX_SPEED = self.max_speed
            self.road.vehicles.append(new_vehicle)

        return obs, info

class CurriculumCallback(BaseCallback):
    def __init__(self, scheduler, hparam_callback, verbose=0):
        super().__init__(verbose)
        self.scheduler = scheduler
        self.hparam_callback = hparam_callback

    def _on_training_start(self):
        self.hparam_callback.init_callback(self.model)
        self.hparam_callback.on_training_start(locals(), globals())

    def _on_step(self):
        timestep = self.num_timesteps
        if timestep % 1000 == 0:
            self.scheduler.update(timestep)
            new_config = self.scheduler.get_env_config()
            self.training_env.envs[0].unwrapped.set_config(new_config)
            self.training_env.reset()
            print(f"[Curriculum] Timestep {timestep} - Vehicles: {new_config['vehicles_count']}")
        self.hparam_callback.on_step()
        return True

    def _on_training_end(self):
        self.hparam_callback.on_training_end()

# Register the environment
gymnasium.register(id="custom-merge-v0", entry_point="__main__:CustomMergeEnv")

# Instantiate environment
env = gymnasium.make("custom-merge-v0", render_mode="rgb_array", config={
    "other_vehicles_type": config_merge.other_vehicles_type,
    "observation": {
        "type": config_merge.observation_type,
        "vehicles_count": config_merge.observation_vehicles_count,
        "features": config_merge.observation_features
    },
    "action": {"type": config_merge.action_type},
    "vehicles_count": config_merge.vehicles_count,
    "scaling": 4.0,
    "screen_width": 800,
    "screen_height": 200
})

# Model configuration
policy_kwargs = dict(net_arch=[128, 128], activation_fn=nn.ReLU)
model = DQN("MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=config_merge.learning_rate,
            buffer_size=config_merge.buffer_size,
            learning_starts=config_merge.learning_starts,
            batch_size=config_merge.batch_size,
            gamma=config_merge.gamma,
            train_freq=config_merge.train_frequency,
            exploration_fraction=config_merge.exploration_fraction,
            target_update_interval=config_merge.target_update_interval,
            tensorboard_log="./DQN_Merge_Model_Curriculum_tensorboard",
            verbose=1)

scheduler = curriculum_scheduler.CurriculumScheduler()
callback = CurriculumCallback(scheduler=scheduler, hparam_callback=config_tensorboard.HParamCallback())

model.learn(
    total_timesteps=config_merge.total_timesteps,
    progress_bar=True,
    callback=callback
)

model.save("DQN_Merge_Model_Curriculum")

# Evaluation
model = DQN.load("DQN_Merge_Model_Curriculum", env=env)
obs, info = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
