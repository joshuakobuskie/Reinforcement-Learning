class CurriculumScheduler:
    def __init__(self):
        self.stage = 0

    def update(self, timestep):
        if timestep < 50000:
            self.stage = 0
        elif timestep < 100000:
            self.stage = 1
        elif timestep < 150000:
            self.stage = 2
        else:
            self.stage = 3

    def get_env_config(self):
        base_vehicle_count = 5
        if self.stage == 0:
            return {"initial_min_speed": 0.001, "initial_max_speed": 15, "stage": 0}
        elif self.stage == 1:
            return {"initial_min_speed": 0.001, "initial_max_speed": 25, "stage": 1}
        elif self.stage == 2:
            return {"initial_min_speed": 0.001, "initial_max_speed": 35, "stage": 2}
        elif self.stage == 3:
            return {"initial_min_speed": 0.001, "initial_max_speed": 35, "stage": 3}
