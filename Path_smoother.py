import math
import hyper as hyp
import time

class PathSmoother:
    def __init__(self):
        self.path = []
        self.cones = []

    def load_data(self, path, cones):
        self.path = path
        self.cones = cones

    def smooth_path(self):
        time_stamp = time.time()
        step_size = hyp.SMOOTHING_STEP_SIZE
        for _ in range(hyp.SMOOTHING_ITERATIONS):
            smoothed_path = []
            for i in range(len(self.path)):
                force = self.cal_force(i)
                new_x = self.path[i][0] + force[0] * step_size
                new_y = self.path[i][1] + force[1] * step_size
                smoothed_path.append((new_x, new_y))
            self.path = smoothed_path
            step_size *= hyp.SMOOTHING_STEP_DECAY
        period = time.time() - time_stamp
        return self.path, period

    def cal_force(self, index):
        if index == 0 or index == len(self.path) - 1:
            return (0.0, 0.0)
        cone_force = (0.0, 0.0)
        for cone in self.cones:
            dx = cone.x - self.path[index][0]
            dy = cone.y - self.path[index][1]
            dist = math.sqrt(dx**2 + dy**2)
            if dist > hyp.SMOOTHING_CONE_DIST_THRESHOLD:
                continue
            decay = math.exp(- dist ** 2 * hyp.SMOOTHING_CONE_DIST_DECAY) * cone.confidence
            cone_force = (cone_force[0] + decay * (-dx / dist), cone_force[1] + decay * (-dy / dist))

        smooth_force = (0.0, 0.0)
        if index > 0:
            prev_dx = self.path[index][0] - self.path[index - 1][0]
            prev_dy = self.path[index][1] - self.path[index - 1][1]
            dist = math.sqrt(prev_dx**2 + prev_dy**2)
            smooth_force = (smooth_force[0] - prev_dx, smooth_force[1] - prev_dy)
        if index < len(self.path) - 1:
            next_dx = self.path[index][0] - self.path[index + 1][0]
            next_dy = self.path[index][1] - self.path[index + 1][1]
            dist = math.sqrt(next_dx**2 + next_dy**2)
            smooth_force = (smooth_force[0] - next_dx, smooth_force[1] - next_dy)

        forceX = cone_force[0] * hyp.SMOOTHING_CONE_INFLUENCE + smooth_force[0] * (1.0 - hyp.SMOOTHING_CONE_INFLUENCE)
        forceY = cone_force[1] * hyp.SMOOTHING_CONE_INFLUENCE + smooth_force[1] * (1.0 - hyp.SMOOTHING_CONE_INFLUENCE)
        return (forceX, forceY)