import random
from itertools import product
import numpy as np
from location.CModel import *
from location.PModel import *

np.random.seed(0)
# 求解器设置
solver_list = listSolvers(onlyAvailable=True)
print(solver_list)  # 可用求解器输出

""" 覆盖模型 """
# num_points = 5  # I: set of the demand points
# num_facilities = 10  # J: set of possible facility location_test
# setup_cost = [3, 2, 3, 1, 3, 3, 4, 3, 2, 4]  # f: cost of locate each facility
# cover = np.random.randint(2, size=(num_points, num_facilities))  # a：facility at j can cover point i
# mp = CoveringModel(num_points=num_points,
#                    num_facilities=num_facilities,
#                    setup_cost=setup_cost,
#                    cover=cover,
#                    solver=GUROBI()).prob_solve()

""" 最大覆盖模型 """
# num_points = 10  # I: set of the demand points
# num_located = 5  # P: number of facility
# num_people = np.random.randint(6, size=num_points)  # h
# cover = np.random.randint(2, size=(num_points, num_points))  # a
# mp2 = MaximumCoveringModel(num_points=num_points,
#                            num_located=num_located,
#                            num_people=num_people,
#                            cover=cover,
#                            solver=GUROBI()).prob_solve()
#
# """ p center """
# num_points = 10
# points = [(random.random(), random.random()) for i in range(num_points)]  # I:collection of demand point location_test
# num_located = 2  # P: number of located facility in the end
# cartesian_prod = list(product(range(num_points), range(num_points)))
# mp3 = PCenter(num_points=num_points,
#               num_located=num_located,
#               cartesian_prod=cartesian_prod,
#               cover=points,
#               solver=GUROBI()).prob_solve()
#
# """ p dispersion """
# num_points = 10
# points = [(random.random(), random.random()) for i in range(num_points)]
# num_located = 5  # P: number of located facility in the end
# cartesian_prod = list(product(range(num_points), range(num_points)))
# mp4 = PDispersion(num_points=num_points,
#                   num_located=num_located,
#                   cartesian_prod=cartesian_prod,
#                   cover=points,
#                   solver=GUROBI()).prob_solve()
#
# """ p median """
# num_points = 10
# points = [(random.random(), random.random()) for i in range(num_points)]
# num_located = 2  # P: number of located facility in the end
# num_people = np.random.randint(6, size=num_points)  # h
# cartesian_prod = list(product(range(num_points), range(num_points)))
# mp5 = PMedian(num_points=num_points,
#               num_located=num_located,
#               cartesian_prod=cartesian_prod,
#               num_people=num_people,
#               cover=points,
#               solver=GUROBI()).prob_solve()
#
# """ MEXCLP """
# num_points = 6
# num_located = 2
# unprob_rate = 0.2
# demand = np.random.randint(1, num_points*0.8, size=num_points)
# cover = np.random.randint(2, size=(num_points, num_points))
# mp6 = MaximumExpectedCoverageLocation(num_points=num_points,
#                                       num_located=num_located,
#                                       demand=demand,
#                                       unprob_rate=unprob_rate,
#                                       cover=cover,
#                                       solver=GUROBI()).prob_solve()

num_points = 10
num_hubs = 3
PC, PT, PD = 1.0, 0.75, 1.25
W = np.random.randint(1, 100, size=(num_points, num_points))
X = random.sample(range(100), num_points)
Y = random.sample(range(100), num_points)
mp7 = PHub(num_points=num_points,
           num_hubs=num_hubs,
           collect_cost=PC,
           transfer_cost=PT,
           distribution_cost=PD,
           x_lat=X,
           y_lon=Y,
           cover=W,
           solver=GUROBI()).prob_solve()
