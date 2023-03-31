from pulp import *
from math import sqrt


def compute_distance(loc1, loc2):
    dx = loc1[0] - loc2[0]
    dy = loc1[1] - loc2[1]
    return sqrt(dx * dx + dy * dy)


class Locate:
    def __init__(self, num_points, cover, solver):
        self.num_points = num_points
        self.cover = cover
        self.solver = solver


class Flow:
    def __init__(self, num_path, num_vector, path_vector, solver):
        self.num_path = num_path
        self.num_vector = num_vector
        self.path_vector = path_vector
        self.solver = solver
