import gymnasium
import highway_env
from matplotlib import pyplot as plt
import pprint
import config
import numpy as np
from highway_env.envs.merge_env import MergeEnv

def euclidian_distance(pos_1, pos_2):
    return np.linalg.norm(np.array(pos_1) - np.array(pos_2))


# Custom environment with modified reward function
class CustomMergeEnv(MergeEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
            
        #Set vehicle start speed between 5 and 15 m/s
        #Set min of 5 and max of 15 m/s 
        for vehicle in self.road.vehicles:
            vehicle.speed = np.random.randint(config.initial_min_speed, config.initial_max_speed)
            vehicle.MIN_SPEED = config.min_speed
            vehicle.MAX_SPEED = config.max_speed

    def _reward(self, action):

        reward = 0.0
        
        #Collisions
        if self.vehicle.crashed:
            reward += config.w1 * -1

        #Speed
        reward += config.w2 * ((self.vehicle.MAX_SPEED - np.sqrt((self.vehicle.speed - self.vehicle.MAX_SPEED)**2))/self.vehicle.MAX_SPEED)

        #Rear/Lateral        
        for vehicle in self.road.vehicles:
            if vehicle != self.vehicle:
                distance = euclidian_distance(self.vehicle.position, vehicle.position)
                #Lateral
                if vehicle.lane != self.vehicle.lane:
                    if distance < config.safety_distance:
                        reward += config.w3 * (-1/distance)
                #Rear
                else:
                    if distance < config.safety_distance:
                        reward += config.w4 * (-1/distance)

        return reward

# Register the custom environment
gymnasium.register(id="custom-merge-v0", entry_point="__main__:CustomMergeEnv")

#Cannot reduce lane count to match graphics
env = gymnasium.make("custom-merge-v0", render_mode="rgb_array", config={"other_vehicles_type": config.other_vehicles_type,
                                                                  "observation": {"type": config.observation_type, "vehicles_count": config.observation_vehicles_count, "features": config.observation_features},
                                                                  "action": {"type": config.action_type}})

#Set vehicle start speed between 5 and 15 m/s
#Set min of 5 and max of 15 m/s
# for vehicle in env.unwrapped.road.vehicles:
#     print("INITIAL\nVehicle: {}, Initial Speed: {}, Min Speed: {}, Max Speed: {}".format(vehicle, vehicle.speed, vehicle.MIN_SPEED, vehicle.MAX_SPEED))
    # vehicle.speed = np.random.randint(config.initial_min_speed, config.initial_max_speed)
    # vehicle.MIN_SPEED = config.min_speed
    # vehicle.MAX_SPEED = config.max_speed
    # print("UPDATED\nVehicle: {}, Initial Speed: {}, Min Speed: {}, Max Speed: {}".format(vehicle, vehicle.speed, vehicle.MIN_SPEED, vehicle.MAX_SPEED))

env.reset()

#Save start position
start_pos = np.copy(env.unwrapped.vehicle.position)
done = False

#Stop on crash or distance greater than 370
while not done and euclidian_distance(start_pos, env.unwrapped.vehicle.position) < config.max_distance:
    print(euclidian_distance(start_pos, env.unwrapped.vehicle.position))
    action = env.unwrapped.action_type.actions_indexes["IDLE"]
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    #Confirmed working as intended: 3 objects with only x,y,vx,vy
    print(obs)

pprint.pprint(env.unwrapped.config)