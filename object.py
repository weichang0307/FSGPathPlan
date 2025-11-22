

BLUE = 'blue'
YELLOW = 'yellow'
ORANGE = 'orange'


class Cone:
    def __init__(self, x, y, color, confidence=1.0):
        self.x = x
        self.y = y
        self.color = color
        self.confidence = confidence