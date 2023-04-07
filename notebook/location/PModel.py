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
                if self.x[i].varValue == 1:
                    selected.append(i)
                    selected_points.append(self.cover[i])
                else:
                    unselected_points.append(self.cover[i])
            if self.name == 'p-center' or self.name == 'p-median':
                for i in range(self.num_points):
                    for j in range(self.num_points):
                        if self.y[i][j].varValue == 1:
                            assigned.append((i, j))

        print("Selected positions =", selected)
        if self.name == 'p-center' or self.name == 'p-median':
            print("Assigned relationships = ", assigned)
            print("Minimum total distance = ", value(prob.objective))
        elif self.name == 'p-dispersion':
            print("Minimum minimum distance between two points = ", value(prob.objective))

        return selected, selected_points, unselected_points

    def draw_plot(self, selected_points, unselected_points):
        plt.figure(figsize=(5, 5))
        plt.title(self.name + ' Problem(P=' + str(self.num_located) + ',I=' + str(self.num_points) + ')')
        plt.scatter(*zip(*selected_points), c='Red', marker=',', s=15, label='Ponit')
        plt.scatter(*zip(*unselected_points), c='Orange', marker='o', s=20, label='Facility')
        if self.name == 'p-center' or self.name == 'p-median':
            for i in range(self.num_points):
                for j in range(self.num_points):
                    if self.y[i][j].varValue == 1:
                        pts = [self.cover[i], self.cover[j]]
                        plt.plot(*zip(*pts), c='Black', linewidth=0.5)
        plt.grid(False)
        plt.legend(loc='best', fontsize=10)
        plt.show()


class PCenter(PModel):

    def __init__(self, num_points, cover, solver, num_located, cartesian_prod):
        super().__init__(num_points, cover, solver, num_located, cartesian_prod)
        self.x = None
        self.y = None
        self.name = 'p-center'

    def prob_solve(self):
        global selected_points
        distance = {(i, j): compute_distance(self.cover[i], self.cover[j]) for i, j in self.cartesian_prod}
        # use distance[i,j] to get the num
        # Create a new model
        prob = LpProblem("p-Cernter_Problem", LpMinimize)

        # Create variables
        Zones = list(range(self.num_points))
        x = LpVariable.dicts("Select", Zones, cat="Binary")  # X
        y = LpVariable.dicts("Assign", (Zones, Zones), cat="Binary")  # Y
        self.y = y
        self.x = x
        # Set objective
        prob += lpSum([[y[i][j] * distance[i, j] for i in range(self.num_points)] for j in
                       range(self.num_points)])  # Minimum total distance

        # Add constraints
        prob += (lpSum([x[i] for i in range(self.num_points)])) == self.num_located  # Fixed total number of facilities

        for i in range(self.num_points):
            prob += (lpSum(
                [y[i][j] for j in range(self.num_points)])) == 1  # Each point only corresponds to one facility

        for j in range(self.num_points):
            for i in range(self.num_points):
                prob += y[i][j] <= x[j]  # Assign before locate; Points can only be assigned to facilities

        selected, selected_points, unselected_points = PModel.show_result(self, prob)
        return self.y, selected, selected_points, unselected_points


class PDispersion(PModel):
    def __init__(self, num_points, cover, solver, num_located, cartesian_prod):
        super().__init__(num_points, cover, solver, num_located, cartesian_prod)
        self.x = None
        self.name = 'p-dispersion'

    def prob_solve(self):
        distance = {(i, j): compute_distance(self.cover[i], self.cover[j])
                    for i, j in self.cartesian_prod
                    if i != j}
        # use distance[i,j] to get the num
        M = 100
        # Create a new model
        prob = LpProblem("p-Dispersion_Problem", LpMaximize)
        # Create variables
        Zones = list(range(self.num_points))
        x = LpVariable.dicts("Select", Zones, cat="Binary")  # X
        self.x = x
        D_min = LpVariable("min_Distance", lowBound=0, cat="Continuous")  # D_min

        # Set objective
        prob += D_min

        # Add constraints
        prob += (lpSum([x[i] for i in range(self.num_points)])) == self.num_located  # Fixed total number of facilities
        for i, j in self.cartesian_prod:
            if i != j:
                prob += (2 - x[i] - x[j]) * M + distance[i, j] >= D_min

        selected, selected_points, unselected_points = PModel.show_result(self, prob)
        return selected, selected_points, unselected_points


class PMedian(PModel):
    def __init__(self, num_people, num_points, cover, solver, num_located, cartesian_prod):
        super().__init__(num_points, cover, solver, num_located, cartesian_prod)
        self.x = None
        self.num_people = num_people
        self.name = 'p-median'
        self.y = None

    def prob_solve(self):
        distance = {(i, j): compute_distance(self.cover[i], self.cover[j]) for i, j in self.cartesian_prod}

        # Create a new model
        prob = LpProblem("p-Median_Problem", LpMinimize)

        # Create variables
        Zones = list(range(self.num_points))
        x = LpVariable.dicts("Select", Zones, cat="Binary")  # X
        y = LpVariable.dicts("Assign", (Zones, Zones), cat="Binary")  # Y
        self.x = x
        self.y = y
        # Set objective
        prob += lpSum([[(y[i][j] * distance[i, j] * self.num_people[i]) for i in range(self.num_points)] for j in
                       range(self.num_points)])  # Minimum total distance

        # Add constraints
        prob += (lpSum([x[i] for i in range(self.num_points)])) == self.num_located  # Fixed total number of facilities
        for i in range(self.num_points):
            prob += (lpSum(
                [y[i][j] for j in range(self.num_points)])) == 1  # Each point only corresponds to one facility

        for j in range(self.num_points):
            for i in range(self.num_points):
                prob += y[i][j] <= x[j]  # Assign before locate; Points can only be assigned to facilities

        selected, selected_points, unselected_points = PModel.show_result(self, prob)
        return self.y, selected, selected_points, unselected_points
        # PModel.draw_plot(self, selected_points, unselected_points)


class PHub(Locate):
    def __init__(self, num_points, cover, solver, num_hubs, collect_cost, transfer_cost, distribution_cost, x_lat,
                 y_lon):
        super().__init__(num_points, cover, solver)
        self.num_hubs = num_hubs
        self.PC = collect_cost
        self.PT = transfer_cost
        self.PD = distribution_cost
        self.X = x_lat
        self.Y = y_lon
        self.Oi = np.sum(self.cover, 1)  # 节点的起始货流量
        self.Di = np.sum(self.cover, 0)  # 节点的终止货流量

    def draw_plot(self, A):
        N = list(range(len(self.X)))
        H = list(set(A))
        for k1 in range(len(H)):
            for k2 in range(k1, len(H)):
                i = H[k1]
                j = H[k2]
                plt.plot([self.X[i], self.X[j]], [self.Y[i], self.Y[j]], color='g')
        for i in set(N) - set(H):
            plt.plot([self.X[i], self.X[A[i]]], [self.Y[i], self.Y[A[i]]], color='k', linewidth=0.5)
            plt.plot(self.X[i], self.Y[i], 'ko')
        for i in H:
            plt.plot(self.X[i], self.Y[i], 'ro', markersize=10)
        plt.show()

    def prob_solve(self):
        C = zip(self.X, self.Y)
        C = [[a, b] for (a, b) in C]
        C = pdist(C, metric='euclidean')
        C = (squareform(C) / 1000).astype(np.int_)
        NN = self.cover.shape[0]
        N = range(NN)
        # 问题定义
        prob = LpProblem("P_HUB", LpMinimize)
        x = {(i, j): LpVariable(cat=LpBinary, name="x_{0}_{1}".format(i, j)) for i in N for j in N}
        y = {(i, j, k): LpVariable(cat=LpContinuous, lowBound=0, name="y_{0}_{1}_{2}".format(i, j, k)) for i in N for j
             in N for k in N}
        # 目标函数
        prob += lpSum(self.PC * C[i][k] * x[i, k] * self.Oi[i] for i in N for k in N) + lpSum(
            self.PT * C[k][h] * y[i, k, h] for i in N for k in N for h in N) + lpSum(
            self.PD * C[k][i] * x[i, k] * self.Di[i] for i in N for k in N)
        # 约束条件
        for i in N:
            prob += lpSum(x[i, j] for j in N) == 1
            for k in N:
                prob += x[i, k] <= x[k, k]
                prob += (lpSum(y[i, k, j] for j in N) - lpSum(y[i, j, k] for j in N)) == (
                        self.Oi[i] * x[i, k] - lpSum(self.cover[i][j] * x[j, k] for j in N))
                prob += lpSum(y[i, k, j] for j in N) <= self.Oi[i] * x[i, k]
        prob += lpSum(x[k, k] for k in N) == self.num_hubs
        prob.solve(self.solver)
        print("Status:", LpStatus[prob.status])
        if LpStatus[prob.status] == "Optimal":
            A = [0] * NN
            for k in N:
                if x[k, k].varValue > 0.5:
                    for i in N:
                        if x[i, k].varValue >= 0.5:
                            A[i] = k
            A = np.array(A)
            P = set(A)
            print("Selected hubs =", list(P))
            print("Minimum total cost = ", value(prob.objective))
            for i in P:
                print("p-Hub {}:".format(i), end="")
                pi = np.where(A == i)
                print(list(pi[0]))
            return A, self.X, self.Y, list(P)
            # PHub.draw_plot(self, A)
