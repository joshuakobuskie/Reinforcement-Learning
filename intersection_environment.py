import gymnasium
import highway_env
import time
import config_intersection
import numpy as np
import random
import torch
import torch.nn as nn
from stable_baselines3 import DQN
from highway_env.envs.intersection_env import IntersectionEnv
import config_tensorboard
from tqdm import tqdm
import curriculum_scheduler
from stable_baselines3.common.callbacks import BaseCallback
from highway_env.vehicle.behavior import IDMVehicle, AggressiveVehicle, DefensiveVehicle

def euclidian_distance(pos_1, pos_2):
    return np.linalg.norm(np.array(pos_1) - np.array(pos_2))

# Custom environment with modified reward function
class CustomIntersectionEnv(IntersectionEnv):
    def __init__(self, *args, **kwargs):
        self.vehicles_count = config_intersection.vehicles_count
        self.initial_min_speed = config_intersection.initial_min_speed
        self.initial_max_speed = config_intersection.initial_max_speed
        self.min_speed = config_intersection.min_speed
        self.max_speed = config_intersection.max_speed
        self.curriculum_stage = 0
        super().__init__(*args, **kwargs)
        #Set min of 5 and max of 15 m/s 
        for vehicle in self.road.vehicles:
            vehicle.speed = np.random.randint(self.initial_min_speed, self.initial_max_speed)
            vehicle.MIN_SPEED = self.min_speed
            vehicle.MAX_SPEED = self.max_speed

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
                        reward += config_intersection.w4 * (-1/distance)
                #Rear
                else:
                    if distance < config_intersection.safety_distance:
                        reward += config_intersection.w3 * (-1/distance)

        return reward

    def set_config(self, new_config):
        for key, val in new_config.items():
            if hasattr(self, key):
                setattr(self, key, val)
        if "stage" in new_config:
            self.curriculum_stage = new_config["stage"]
    
    #Need to add 370 meter stopping constraint into the env
    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        if euclidian_distance(self.start_pos, self.vehicle.position) >= config_intersection.max_distance:
            done = True
        return obs, reward, done, truncated, info
    
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.start_pos = np.copy(self.vehicle.position)

        # agent_vehicle = self.vehicle
        # agent_vehicle.speed = np.random.uniform(self.initial_min_speed, self.initial_max_speed)
        # agent_vehicle.MIN_SPEED = self.min_speed
        # agent_vehicle.MAX_SPEED = self.max_speed

        # new_vehicles = [agent_vehicle]
        # target_vehicle_count = self.vehicles_count
        # existing_vehicles = [v for v in self.road.vehicles if v is not agent_vehicle]
        # retained_vehicles = existing_vehicles[:target_vehicle_count - 1]
        # additional_needed = max(0, (target_vehicle_count - 1) - len(retained_vehicles))
        # placeholders = [None] * additional_needed
        # all_vehicles = retained_vehicles + placeholders

        # print(all_vehicles)
        # for v in all_vehicles:
        #     if v is not None:
        #         pos = v.position
        #         heading = v.heading
        #         print(pos)
        #         print(heading)
        #     else:
        #         lane = random.choice(self.road.network.lanes_list())
        #         longitudinal = np.random.uniform(0, lane.length)
        #         lateral = 0
        #         pos = lane.position(longitudinal, lateral)
        #         heading = lane.heading_at(longitudinal)
        #         # print(pos)
        #         # print(heading)

        #     # Choose vehicle class based on curriculum
        #     if self.curriculum_stage == 0:
        #         vehicle_cls = IDMVehicle
        #     elif self.curriculum_stage == 1:
        #         vehicle_cls = DefensiveVehicle
        #     elif self.curriculum_stage == 2:
        #         vehicle_cls = AggressiveVehicle
        #     elif self.curriculum_stage == 3:
        #         vehicle_cls = random.choice([IDMVehicle, DefensiveVehicle, AggressiveVehicle])
        #     else:
        #         vehicle_cls = IDMVehicle

        #     new_v = vehicle_cls(
        #         self.road,
        #         position=pos,
        #         heading=heading,
        #         speed=np.clip(np.random.uniform(self.initial_min_speed, self.initial_max_speed),
        #                       self.min_speed, self.max_speed)
        #     )
        #     new_v.MIN_SPEED = self.min_speed
        #     new_v.MAX_SPEED = self.max_speed

        #     new_vehicles.append(new_v)

        # self.road.vehicles = new_vehicles
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
gymnasium.register(id="custom-intersection-v0", entry_point="__main__:CustomIntersectionEnv")

env = gymnasium.make("custom-intersection-v0", render_mode="rgb_array", config={
    "other_vehicles_type": config_intersection.other_vehicles_type,
    "observation": {
        "type": config_intersection.observation_type,
        "vehicles_count": config_intersection.observation_vehicles_count,
        "features": config_intersection.observation_features
    },
    "action": {"type": config_intersection.action_type},
    "scaling": 3.0,
    "screen_width": 800,
    "screen_height": 200
})

#Create model

#######################################
#Uncomment when training a new model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scheduler = curriculum_scheduler.CurriculumScheduler()
callback = CurriculumCallback(scheduler=scheduler, hparam_callback=config_tensorboard.HParamCallback())
# print("Using device:", device)  # Check if GPU is available
# policy_kwargs = dict(net_arch=[128, 128], activation_fn=nn.ReLU)
# model = DQN(
#     "MlpPolicy",
#     env,
#     policy_kwargs=policy_kwargs,
#     learning_rate=config_intersection.learning_rate,
#     tensorboard_log="./DQN_Intersection_Model_Eval_tensorboard/",
#     buffer_size=config_intersection.buffer_size,
#     learning_starts=config_intersection.learning_starts,
#     batch_size=config_intersection.batch_size,
#     gamma=config_intersection.gamma,
#     train_freq=config_intersection.train_frequency,
#     exploration_fraction=config_intersection.exploration_fraction,
#     target_update_interval=config_intersection.target_update_interval,
#     verbose=1,
#     device=device)

# model.learn(
#     total_timesteps=config_intersection.total_timesteps, 
#     progress_bar=True, 
#     callback=config_tensorboard.HParamCallback())
# model.save("DQN_Intersection_Model_Eval")
######################################

model = DQN.load("DQN_Merge_Model_Curriculum", env=env)
# obs, info = env.reset()
# done = False

# while not done:
#     action, next_state = model.predict(obs, deterministic=True)
#     obs, reward, done, truncated, info = env.step(action)
#     env.render()
##### EVALUATE ONLY #############
for stage in range(4):
    print(f"\n--- Evaluating Stage {stage} ---")

    # Set environment stage manually
    scheduler.stage = stage 
    stage_config = scheduler.get_env_config()
    env.unwrapped.set_config(stage_config)

    obs, info = env.reset(seed=stage + 42)
    done = False
    episode_reward = 0
    print(f"Vehicle count: {len(env.unwrapped.road.vehicles)}")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        env.render()  # Shows visual if using render_mode="human" or "rgb_array"
        # Optional: Slow down rendering to observe
        time.sleep(0.05)

    print(f"Total reward for one episode (Stage {stage}): {episode_reward}")
