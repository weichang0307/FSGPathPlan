import Yaw_optimizer
import math
import time
import hyper as hyp
from Yaw_optimizer import YawOptimizer

class PathPlanner:
    def __init__(self):
        self.step_size = hyp.STEP_SIZE
        self.num_steps = hyp.NUM_STEPS
        self.optimizer = YawOptimizer()

    def generate_path(self, start_pos, start_yaw, cones):
        timestamp = time.time()
        path = []
        probs = []
        pos = start_pos
        yaw = start_yaw
        prob = 1.0
        
        for _ in range(self.num_steps):
            self.optimizer.load_data(pos, yaw, cones)
            self.optimizer.optimize_yaw()
            yaw = self.optimizer.yaw
            pos = (pos[0] + math.cos(yaw) * self.step_size, pos[1] + math.sin(yaw) * self.step_size)
            step_prob = self.optimizer.cal_prob()
            prob *= step_prob
            path.append(pos)
            probs.append(prob)
            
        period = time.time() - timestamp
        return path, probs, period