import object as obj
import hyper as hyp
import math


class BoundaryEstimator:
    def __init__(self, cones):
        self.cones = cones
        self.blue_boundary = []
        self.yellow_boundary = []

    def initialize_boundaries(self, blue_boundary=[], yellow_boundary=[]):
        # Initialize the boundary with at least two points
        self.blue_boundary = blue_boundary
        self.yellow_boundary = yellow_boundary

    def estimate_yellow_boundary(self):
        max_prob = -1.0
        while max_prob > 0.01 or max_prob == -1.0:
            max_prob = 0.0
            best_cone = None
            cones = []
            for cone in self.cones:
                if cone.color != obj.YELLOW:
                    continue
                if len(self.yellow_boundary) < 2:
                    raise Exception("Yellow boundary not initialized.")
                px0, py0 = self.yellow_boundary[-2]
                px1, py1 = self.yellow_boundary[-1]
                px2, py2 = cone.x, cone.y
                angle = math.atan2(py2 - py1, px2 - px1) - math.atan2(py1 - py0, px1 - px0)
                if abs(angle) < math.pi / 2:
                    cones.append(cone)
                prob = self.yellow_segment_prob(px1, py1, px2, py2, cones)
                if prob > max_prob:
                    max_prob = prob
                    best_cone = cone
            if best_cone is None:
                break
            self.yellow_boundary.append(best_cone)
            self.cones.remove(best_cone)
    
    def estimate_blue_boundary(self):
        max_prob = -1.0
        while max_prob > 0.01 or max_prob == -1.0:
            max_prob = 0.0
            best_cone = None
            cones = []
            for cone in self.cones:
                if cone.color != obj.BLUE:
                    continue
                if len(self.blue_boundary) < 2:
                    raise Exception("Blue boundary not initialized.")
                px0, py0 = self.blue_boundary[-2]
                px1, py1 = self.blue_boundary[-1]
                px2, py2 = cone.x, cone.y
                angle = math.atan2(py2 - py1, px2 - px1) - math.atan2(py1 - py0, px1 - px0)
                if abs(angle) < math.pi / 2:
                    cones.append(cone)
                prob = self.yellow_segment_prob(px1, py1, px2, py2, cones)
                if prob > max_prob:
                    max_prob = prob
                    best_cone = cone
            if best_cone is None:
                break
            self.blue_boundary.append(best_cone)
            self.cones.remove(best_cone)
    
    def yellow_segment_prob(self, px1, py1, px2, py2, cones):
        sum_w = 0.0
        score = 0.0
        for cone in cones:
            mx, my = (px1 + px2) / 2, (py1 + py2) / 2
            cone_distance = math.sqrt((cone.x - mx) ** 2 + (cone.y - my) ** 2)
            if cone_distance > hyp.BE_DIST_THRESHOLD:
                continue
            w = math.exp(-cone_distance / hyp.BE_DIST_DECAY)
            sum_w += w
            x = (self.point_line_distance((cone.x, cone.y), (px1, py1), (px2, py2)) - hyp.BE_MID_TRACK_WIDTH) ** 2
            if cone.color == obj.YELLOW:
                score -= w * max(hyp.BE_MID_TRACK_WIDTH **2 - x, 0)
            elif cone.color == obj.BLUE:
                if x < hyp.BE_TRACK_WIDTH_DIFF ** 2:
                    x = 0.0
                score -= w * x
        if sum_w == 0:
            return 0.0
        score /= sum_w
        score -= hyp.BE_LENGTH_DECAY * max(math.sqrt((px2 - px1) ** 2 + (py2 - py1) ** 2) - hyp.BE_LENGTH_THRESHOLD, 0)
        return math.exp(score)
    
    def yellow_segment_prob(self, px1, py1, px2, py2, cones):
        sum_w = 0.0
        score = 0.0
        for cone in cones:
            mx, my = (px1 + px2) / 2, (py1 + py2) / 2
            cone_distance = math.sqrt((cone.x - mx) ** 2 + (cone.y - my) ** 2)
            if cone_distance > hyp.BE_DIST_THRESHOLD:
                continue
            w = math.exp(-cone_distance / hyp.BE_DIST_DECAY)
            sum_w += w
            x = (-self.point_line_distance((cone.x, cone.y), (px1, py1), (px2, py2)) - hyp.BE_MID_TRACK_WIDTH) ** 2
            if cone.color == obj.YELLOW:
                score -= w * max(hyp.BE_MID_TRACK_WIDTH **2 - x, 0)
            elif cone.color == obj.BLUE:
                if x < hyp.BE_TRACK_WIDTH_DIFF ** 2:
                    x = 0.0
                score -= w * x
        if sum_w == 0:
            return 0.0
        score /= sum_w
        score -= hyp.BE_LENGTH_DECAY * max(math.sqrt((px2 - px1) ** 2 + (py2 - py1) ** 2) - hyp.BE_LENGTH_THRESHOLD, 0)
        return math.exp(score)
    
    def point_line_distance(self, point, line_pointA, line_pointB):
        vectorAB = (line_pointB[0] - line_pointA[0], line_pointB[1] - line_pointA[1])
        vectorAP = (point[0] - line_pointA[0], point[1] - line_pointA[1])
        num = vectorAB[0] * vectorAP[1] - vectorAB[1] * vectorAP[0]
        den = math.sqrt((vectorAB[1]) ** 2 + (vectorAB[0]) ** 2)
        if den == 0:
            return float('inf')
        return num / den