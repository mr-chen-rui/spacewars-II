import math
import random


type Vector2 = tuple[float, float]
type Input = tuple[bool, bool, bool] # is accelerating, is turning left, is turning right

def normalize(vector : Vector2) -> Vector2:
    length = math.sqrt(pow(vector[0],2) + pow(vector[1],2))
    return (vector[0] / length, vector[1] / length)

class Target:
    instance = None
    position : Vector2
    max_dist = 500

    def __new__(cls):
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance

    def randomize_position(self):
        theta = random.random()
        r = self.max_dist * math.sqrt(random.random())
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        self.position = (x,y)

class Player:
    position : Vector2
    velocity : Vector2
    direction : float
    fuel_used : float

    def __init__(self):
        self.position = (0,0)
        self.velocity = (0,0)
        self.direction = 0.0

    def physics_tick(self, inputs):
        gravitational_force : float = 0.066743 * (100) / math.pow(math.dist((0,0), self.position),2) # Newton's law of Universal Gravitation says F=G(m1*m2)/r^2
        # where masses are in kg and distances are in m. G is the gravitational constant equal to 0.000000000066743.
        # Had to simplify here for the sake of realism (I'm not typing out 10^30)
        gravity_force : Vector2 = normalize((-self.position[0], -self.position[1]))
        gravity_force = (gravitational_force * gravity_force[0], gravitational_force * gravity_force[1])

        self.position = (self.position[0] + gravity_force[0], self.position[1] + gravity_force[1])

    def proximity_to(self, target : Vector2) -> float:
        return math.sqrt(pow(target[0], 2) + pow(target[1], 2))