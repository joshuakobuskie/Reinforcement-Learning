import gymnasium
import highway_env
import time
import config_merge
import numpy as np
import random
import torch
import torch.nn as nn
from stable_baselines3 import DQN
from highway_env.envs.merge_env import MergeEnv
import config_tensorboard
from tqdm import tqdm
import curriculum_scheduler
from stable_baselines3.common.callbacks import BaseCallback
from highway_env.vehicle.behavior import IDMVehicle, AggressiveVehicle, DefensiveVehicle

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

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.start_pos = np.copy(self.vehicle.position)

        ego = self.vehicle
        ego.speed = np.random.uniform(self.initial_min_speed, self.initial_max_speed)
        ego.MIN_SPEED = self.min_speed
        ego.MAX_SPEED = self.max_speed

        new_vehicles = [ego]

        for v in self.road.vehicles:
            if v is ego:
                continue

            # Choose behavior class based on curriculum
            if self.curriculum_stage == 0:
                vehicle_cls = IDMVehicle
            elif self.curriculum_stage == 1:
                vehicle_cls = DefensiveVehicle
            elif self.curriculum_stage == 2:
                vehicle_cls = AggressiveVehicle
            elif self.curriculum_stage == 3:
                vehicle_cls = random.choice([IDMVehicle, DefensiveVehicle, AggressiveVehicle])
            else:
                vehicle_cls = IDMVehicle  # fallback

            new_v = vehicle_cls(
                self.road,
                position=v.position,
                heading=v.heading,
                speed=np.clip(np.random.uniform(self.initial_min_speed, self.initial_max_speed),
                              self.min_speed, self.max_speed)
            )
            new_v.MIN_SPEED = self.min_speed
            new_v.MAX_SPEED = self.max_speed

            new_vehicles.append(new_v)

        self.road.vehicles = new_vehicles
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
            print(f"[Curriculum] Timestep {timestep} - Stage: {new_config["stage"]}")
        self.hparam_callback.on_step()
        return True

    def _on_training_end(self):
        self.hparam_callback.on_training_end()

# Register the custom environment
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
# policy_kwargs = dict(net_arch=[64, 64], activation_fn=nn.ReLU)
# model = DQN("MlpPolicy",
#             env,
#             policy_kwargs=policy_kwargs,
#             learning_rate=config_merge.learning_rate,
#             buffer_size=config_merge.buffer_size,
#             learning_starts=config_merge.learning_starts,
#             batch_size=config_merge.batch_size,
#             gamma=config_merge.gamma,
#             train_freq=config_merge.train_frequency,
#             exploration_fraction=config_merge.exploration_fraction,
#             target_update_interval=config_merge.target_update_interval,
#             tensorboard_log="./DQN_Merge_Model_Curriculum_tensorboard",
#             verbose=1)

# scheduler = curriculum_scheduler.CurriculumScheduler()
# callback = CurriculumCallback(scheduler=scheduler, hparam_callback=config_tensorboard.HParamCallback())

# model.learn(
#     total_timesteps=config_merge.total_timesteps,
#     progress_bar=True,
#     callback=callback
# )

# model.save("DQN_Merge_Model_Curriculum")

# Evaluation
model = DQN.load("DQN_Merge_Model_Curriculum", env=env)
# obs, info = env.reset()
# done = False
# while not done:
#     action, _ = model.predict(obs, deterministic=True)
#     obs, reward, done, truncated, info = env.step(action)
#     env.render()
model = DQN.load("DQN_Merge_Model_Curriculum")

# Loop through all curriculum stages

for stage in range(4):
    print(f"\n--- Evaluating Stage {stage} ---")

    # Set environment stage manually

    env.unwrapped.set_config({
            "stage": stage,
            "initial_min_speed": config_merge.initial_min_speed,
            "initial_max_speed": config_merge.initial_max_speed,
    })

    obs, info = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        env.render()  # Shows visual if using render_mode="human" or "rgb_array"
        print(obs)

        # Optional: Slow down rendering to observe
        time.sleep(0.05)

    print(f"Total reward (Stage {stage}): {episode_reward}")
