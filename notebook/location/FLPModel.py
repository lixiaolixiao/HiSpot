from location import *
from location import Locate
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform


class PModel(Locate):
    def __init__(self, num_points, cover, solver, num_located, cartesian_prod):
        super().__init__(num_points, cover, solver)
        self.name = None
        self.num_located = num_located
        self.cartesian_prod = cartesian_prod

    def show_result(self, prob):
        global selected, selected_points, unselected_points, assigned
        prob.solve(self.solver)
        print("Status:", LpStatus[prob.status])
        if LpStatus[prob.status] == "Optimal":
            selected, selected_points, unselected_points, assigned = [], [], [], []
            for i in range(self.num_points):
                if self.x[i].varValue > 0:
                    selected.append(i)
                    selected_points.append(self.cover[i])
                else:
                    unselected_points.append(self.cover[i])
            if self.name == 'p-center' or self.name == 'p-median' or self.name == 'CFLP' or self.name == 'UFLP':
                for i in range(self.num_points):
                    for j in range(self.num_points):
                        if self.y[i][j].varValue > 0:
                            assigned.append((i, j))

        print("Selected positions =", selected)
        if self.name == 'p-center' or self.name == 'p-median':
            print("Assigned relationships = ", assigned)
            print("Minimum total distance = ", value(prob.objective))
        elif self.name == 'p-dispersion':
            print("Minimum minimum distance between two points = ", value(prob.objective))
        elif self.name == 'CFLP' or self.name == 'UFLP':
            print("Costs = ", value(prob.objective))
        return selected_points, unselected_points

    def draw_plot(self, selected_points, unselected_points, plot=False):
        if plot:
            plt.figure(figsize=(5, 5))
            plt.title(self.name + ' Problem(P=' + str(self.num_located) + ',I=' + str(self.num_points) + ')')
            plt.scatter(*zip(*selected_points), c='Red', marker=',', s=15, label='Facility')
            plt.scatter(*zip(*unselected_points), c='Orange', marker='o', s=20, label='People')
            if self.name == 'p-center' or self.name == 'p-median' or self.name == 'CFLP' or self.name == 'UFLP':
                for i in range(self.num_points):
                    for j in range(self.num_points):
                        if self.y[i][j].varValue > 0:
                            pts = [self.cover[i], self.cover[j]]
                            plt.plot(*zip(*pts), c='Black', linewidth=0.5)

            plt.grid(False)
            plt.legend(loc='best', fontsize=10)
            plt.show()


class CFLP(PModel):
    def __init__(self, num_people, num_points, cover, solver, num_located, cartesian_prod, demand, capacity, cost):
        super().__init__(num_points, cover, solver, num_located, cartesian_prod)
        self.x = None
        self.demand = demand
        self.capacity = capacity
        self.cost = cost
        self.num_people = num_people
        self.name = 'CFLP'
        self.y = None

    def prob_solve(self):
        distance = {(i, j): compute_distance(self.cover[i], self.cover[j]) for i, j in self.cartesian_prod}

        # Create a new model
        prob = LpProblem("CFLP", LpMinimize)

        # Create variables
        Zones = list(range(self.num_points))
        x = LpVariable.dicts("Select", Zones, cat="Binary")  # X
        y = LpVariable.dicts("Assign", (Zones, Zones), lowBound=0, upBound=1, cat="Continuous")  # Y: Distribution received from each facility
        self.x = x
        self.y = y

        # Set objective
        prob += lpSum([[(y[i][j] * distance[i, j] * self.num_people[i]) for i in range(self.num_points)]
                       for j in range(self.num_points)]) + \
                lpSum([(x[i] * self.cost[i] for i in range(self.num_points))])    # Minimum cost including shippingcost and establishment cost

        # Add constraints
        prob += (lpSum([x[i] for i in range(self.num_points)])) == self.num_located  # Fixed total number of facilities
        for i in range(self.num_points):
            prob += (lpSum([y[i][j] for j in range(self.num_points)])) == 1     #The total amount received from each facility is 1

        for j in range(self.num_points):
            prob += (lpSum([y[i][j] * self.demand[i] for i in range(self.num_points)])) <= self.capacity[j] * x[j]  # Demands need to be under the sum of capacity

        for j in range(self.num_points):
            for i in range(self.num_points):
                prob += y[i][j] <= x[j]  # Assign before locate; Points can only be assigned to facilities

        selected_points, unselected_points = PModel.show_result(self, prob)
        PModel.draw_plot(self, selected_points, unselected_points)

        return y, selected, selected_points, unselected_points

class UFLP(PModel):
    def __init__(self, num_people, num_points, cover, solver, num_located, cartesian_prod, demand, cost):
        super().__init__(num_points, cover, solver, num_located, cartesian_prod)
        self.x = None
        self.demand = demand
        self.cost = cost
        self.num_people = num_people
        self.name = 'UFLP'
        self.y = None

    def prob_solve(self):
        distance = {(i, j): compute_distance(self.cover[i], self.cover[j]) for i, j in self.cartesian_prod}

        # Create a new model
        prob = LpProblem("CFLP", LpMinimize)

        # Create variables
        Zones = list(range(self.num_points))
        x = LpVariable.dicts("Select", Zones, cat="Binary")  # X
        y = LpVariable.dicts("Assign", (Zones, Zones), cat="Binary")  # Y
        self.x = x
        self.y = y
        # Set objective
        prob += lpSum([[(y[i][j] * distance[i, j] * self.num_people[i]) for i in range(self.num_points)]
                       for j in range(self.num_points)]) + \
                lpSum([(x[i] * self.cost[i] for i in range(self.num_points))])    # Minimum cost including shippingcost and establishment cost

        # Add constraints
        prob += (lpSum([x[i] for i in range(self.num_points)])) <= self.num_located     # Under than total number of facilities
        for i in range(self.num_points):
            prob += (lpSum([y[i][j] for j in range(self.num_points)])) == 1     # Each point only corresponds to one facility

        for j in range(self.num_points):
            for i in range(self.num_points):
                prob += y[i][j] <= x[j]  # Assign before locate; Points can only be assigned to facilities

        selected_points, unselected_points = PModel.show_result(self, prob)
        PModel.draw_plot(self, selected_points, unselected_points)

        return y, selected, selected_points, unselected_points