import highway_env
max_iters = 100

#Paper configurations
#Each vehicle was randomly generated within the specified range beneath the vehicles, and the initial velocity was also randomly set between 5 to 15 m/s.
initial_min_speed = 5
initial_max_speed = 15

#HDV were implemented using Intelligent Driver Model (IDM) [9] and Minimizing Overall Braking Induced by Lane change (MOBIL)
#IDMVehcile uses the IDM for longitudinal movement and MOBIL for lateral movement
vehicles_type = highway_env.vehicle.behavior.IDMVehicle


#Receives observations of up to three obstacles or vehicles in its vicinity
#Set observation limit to 3 vehicles
observation_type =  "Kinematics"
observation_vehicles_count = 3

#View only longitudinal coordinate, lateral coordinate, longitudinal velocity, and lateral velocity
observation_features = ["x", "y", "vx", "vy"]

#An episode would terminate if a collision occurred or the ego vehicle passed a distance of 370m
#Maximum distance of simulation
max_distance = 370

#can perform five actions (Lane left, Idle, Lane right, faster, slower) through acceleration and steering control.
action_type = "DiscreteMetaAction"

#corresponding action space
# ACTIONS_ALL = {
#         0: 'LANE_LEFT',
#         1: 'IDLE',
#         2: 'LANE_RIGHT',
#         3: 'FASTER',
#         4: 'SLOWER'
#     }


# The speed range for each vehicle is limited to 5-15 m/s.
min_speed = 5
max_speed = 15

#defaults from the highway env
action = {'type': 'DiscreteMetaAction'}
centering_position = [0.3, 0.5]
collision_reward = -1
controlled_vehicles = 1
duration = 40
ego_spacing = 2
high_speed_reward = 0.4
initial_lane_id = None
lane_change_reward = 0
lanes_count = 4
manual_control = False
normalize_reward = True
observation = {'type': 'Kinematics'}
offroad_terminal = False
offscreen_rendering = True
other_vehicles_type = 'highway_env.vehicle.behavior.IDMVehicle'
policy_frequency = 1
real_time_rendering = False
render_agent = True
reward_speed_range = [20, 30]
right_lane_reward = 0.1
scaling = 5.5
screen_height = 150
screen_width = 600
show_trajectories = False
simulation_frequency = 15

vehicles_density = 1