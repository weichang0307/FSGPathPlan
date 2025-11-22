import numpy as np
import math
import object as obj
import hyper as hyp

'''
This optimizer is used to optimize the yaw angle of next step based on the cones positions.
'''



class YawOptimizer:
    def __init__(self):
        self.step_size = hyp.STEP_SIZE
        self.likelihood_gamma = hyp.LIKELIHOOD_GAMMA
        self.prior_gamma = hyp.PRIOR_GAMMA
        self.dist_decay = hyp.DIST_DECAY
        self.optimize_steps = hyp.OPTIMIZE_STEPS
        self.lr0 = hyp.LR0
        self.lr_min = hyp.LR_MIN
        self.lr = self.lr0
        self.max_cone_dist = 15.0
        self.weight_thresh = math.exp(-self.max_cone_dist * hyp.DIST_DECAY)

        self.ready = False
        self.car_pos = None
        self.car_yaw = None
        self.yaw = None
        self.cones = []
        self.cones_dir = []
        self.cones_dist = []
        self.cones_attention = []
        
    def load_data(self, car_pos, car_yaw, cones):
        self.car_pos = car_pos
        self.car_yaw = car_yaw
        self.yaw = car_yaw
        self.cones = [cone for cone in cones if math.sqrt((cone.x - car_pos[0])**2 + (cone.y - car_pos[1])**2) <= self.max_cone_dist]
        self.process_data()
        
    def process_data(self):
        car_to_cones = [(cone.x - (self.car_pos[0] + self.step_size / 2 * math.cos(self.yaw)), cone.y - (self.car_pos[1] + self.step_size / 2 * math.sin(self.yaw))) for cone in self.cones]
        self.cones_dir = [math.atan2(dy, dx) for dx, dy in car_to_cones]
        self.cones_dist = [math.sqrt(dx**2 + dy**2) for dx, dy in car_to_cones]
        self.cal_attention()
        self.ready = True
        
    def optimize_yaw(self):
        if not self.ready:
            raise Exception("YawOptimizer data not loaded.")
        for _ in range(self.optimize_steps):
            yaw_grad = self.cal_gradient()
            self.yaw += self.lr * yaw_grad
            if math.cos(self.yaw - self.car_yaw) < math.cos(hyp.MAX_YAW_DEVIATION):
                self.yaw = self.car_yaw + math.copysign(hyp.MAX_YAW_DEVIATION, self.yaw - self.car_yaw)
            self.process_data()
            self.lr -= (self.lr0 - self.lr_min) / (self.optimize_steps - 1)
        self.lr = self.lr0
        return self.yaw

    def cal_attention(self):
        self.cones_attention = [math.exp(-max(0, d) * self.dist_decay) for d in self.cones_dist]

    def cal_prob(self):
        prior_prob = self.prior_prob()
        likelihood_prob = self.likelihood_prob()
        prob = prior_prob * likelihood_prob
        return prob

    def cal_gradient(self):
        prior_grad = self.prior_gradient()
        likelihood_grad = self.likelihood_gradient()
        grad = prior_grad + likelihood_grad
        return grad
    
    def prior_prob(self):
        return math.exp(-self.prior_gamma * (self.yaw - self.car_yaw) ** 2)
    
    def likelihood_prob(self):
        logit = self.likelihood_logit()
        prob = 1 / (1 + math.exp(-self.likelihood_gamma * logit))
        return prob

    def likelihood_logit(self):
        logit_total = 0.0
        for index, cone in enumerate(self.cones):
            angle = self.yaw - self.cones_dir[index]
            if math.cos(angle) < 0:
                continue
            if self.cones[index].color == obj.YELLOW:
                logit = math.sin(angle)
            elif self.cones[index].color == obj.BLUE:
                logit = -math.sin(angle)
            else:
                logit = 0.0
            logit_total += logit * self.cones_attention[index] * cone.confidence
        return logit_total
    
    def prior_gradient(self):
        return -self.prior_gamma * 2 * (self.yaw - self.car_yaw)
    
    def likelihood_gradient(self):
        return self.likelihood_logit_gradient() * self.likelihood_gamma / (1 + math.exp(self.likelihood_gamma * self.likelihood_logit()))
    
    def likelihood_logit_gradient(self):
        grad_total = 0.0
        for index, cone in enumerate(self.cones):
            angle = self.yaw - self.cones_dir[index]
            if math.cos(angle) < 0:
                continue
            if self.cones[index].color == obj.YELLOW:
                grad = math.cos(angle)
            elif self.cones[index].color == obj.BLUE:
                grad = -math.cos(angle)
            else:
                grad = 0.0
            grad_total += grad * self.cones_attention[index] * cone.confidence
        return grad_total
    
