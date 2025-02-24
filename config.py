import highway_env
max_iters = 100

#Paper configurations
#IDMVehcile uses the IDM for longitudinal movement and MOBIL for lateral movement
vehicles_type = highway_env.vehicle.behavior.IDMVehicle
initial_min_speed = 5
initial_max_speed = 15

#Includes agent vehicle
#Not what we needed, this limits vehicles not observations
#vehicles_count = 4

#Set observation limit to 3 vehicles
observation_type =  "Kinematics"
observation_vehicles_count = 3

#Maximum distance of simulation
max_distance = 370

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