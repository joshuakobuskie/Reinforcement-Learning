class CurriculumScheduler:
    def __init__(self):
        self.stage = 0

    def update(self, timestep):
        if timestep < 40000:
            self.stage = 0
        elif timestep < 80000:
            self.stage = 1
        elif timestep < 120000:
            self.stage = 2
        else:
            self.stage = 3

    def get_env_config(self):
        base_vehicle_count = 5
        if self.stage == 0:
            return {"initial_min_speed": 0.001, "initial_max_speed": 15, "vehicles_count": base_vehicle_count, "stage": 0}
        elif self.stage == 1:
            return {"initial_min_speed": 0.0001, "initial_max_speed": 25, "vehicles_count": base_vehicle_count + 1, "stage": 1}
        elif self.stage == 2:
            return {"initial_min_speed": 0.00001, "initial_max_speed": 35, "vehicles_count": base_vehicle_count + 2, "stage": 2}
        elif self.stage == 3:
            return {"initial_min_speed": 0.000001, "initial_max_speed": 35, "vehicles_count": base_vehicle_count + 3, "stage": 3}
