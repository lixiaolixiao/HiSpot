from location import *
from location import LocateLRP
import numpy as np

class LRP_capModel(LocateLRP):
    def __init__(self, facility_nodes, demand_nodes, solver, fa_cap, de_demand):
        super().__init__(facility_nodes, demand_nodes, solver)
        self.name = None
        self.fa_cap = fa_cap
        self.de_demand = de_demand

    def show_result(self, prob):
        global selected, selected_facility, unselected, unselected_facility, assigned
        prob.solve(self.solver)
        print("Status:", LpStatus[prob.status])
        if LpStatus[prob.status] == "Optimal":
            selected, selected_facility, unselected, unselected_facility, assigned = [], [], [],[],[]
            for i in range(len(self.facility_nodes)):
                if self.y[i].varValue == 1:
                    selected.append(i)
                    selected_facility.append(self.facility_nodes[i])
                    for j in range(len(self.demand_nodes)):
                        if self.x[i, j].varValue == 1:
                            assigned.append((i, j))
                else:
                    unselected.append(i)
                    unselected_facility.append(self.facility_nodes[i])

        print("Selected facilities =", selected)
        print("Unselected facilities =", unselected)
        print("Assigned relationships = ", assigned)
        print("Minimum total distance = ", value(prob.objective))

        return selected_facility, unselected_facility, assigned

    def draw_plot(self, selected_facility, unselected_facility, assigned):
        plt.figure(figsize=(5, 5))
        plt.title(self.name + ' Problem(Demand=' + str(len(self.demand_nodes)) + ',Facility=' + str(len(self.facility_nodes)) + ')')
        plt.scatter(*zip(*selected_facility), c='Red', marker='*', s=20, label='Served Facility')
        plt.scatter(*zip(*unselected_facility), c='#ff69E1', marker='*', s=30, label='Unserved Facility')
        plt.scatter(*zip(*self.demand_nodes), c='Blue', marker='o', s=20, label='Demand')
        for i in range(len(assigned)):
            pts = [self.facility_nodes[assigned[i][0]], self.demand_nodes[assigned[i][1]]]
            plt.plot(*zip(*pts), c='Orange', linewidth=0.5)
        plt.grid(False)
        plt.legend(loc='best', fontsize=10)
        plt.show()


class LRP_cap(LRP_capModel):

    def __init__(self, facility_nodes, demand_nodes, solver,  fa_cap, de_demand):
        super().__init__(facility_nodes, demand_nodes, solver, fa_cap, de_demand)
        self.x = None
        self.y = None
        self.name = 'LRP_cap'

    def prob_solve(self):
        global selected, selected_facility, unselected, unselected_facility, assigned
        # Compute euclidean distances between each warehouse and demand point.
        num_fa = len(self.facility_nodes)
        num_de = len(self.demand_nodes)
        cost = np.zeros((num_fa, num_de))
        for i in range(num_fa):
            fa = self.facility_nodes[i]
            for j in range(num_de):
                de = self.demand_nodes[j]
                dx = fa[0] - de[0]
                dy = fa[1] - de[1]
                cost[i][j] = sqrt(dx * dx + dy * dy)
        # use cost[i,j] to get the num
        # Create a new model
        prob = LpProblem("Location_Routing_Problem", LpMinimize)

        # Create variables
        set_F = range(0, num_fa)
        set_D = range(0, num_de)
        # Variables
        x = {(i, j):
                 LpVariable(cat=LpBinary, name="x_{0}_{1}".format(i, j))
             for i in set_F for j in set_D}

        y = {(i):
                 LpVariable(cat=LpBinary, name="y_{0}".format(i))
             for i in set_F}
        self.y = y
        self.x = x

        # Add constraints
        for i in set_F:
            prob += (lpSum([x[i, j] * self.de_demand[j] for j in set_D]) <= self.fa_cap[i][1] * y[i])

        for i in set_F:
            prob += (lpSum([x[i, j] * self.de_demand[j] for j in set_D]) >= self.fa_cap[i][0] * y[i])

        constraints_eq = {j: prob.addConstraint(
            LpConstraint(
                e=lpSum(x[i, j] for i in set_F),
                sense=LpConstraintEQ,
                rhs=1,
                name="constraint_eq_{0}".format(j)
            )
        ) for j in set_D}

        # Set objective
        prob += lpSum(x[i, j] * cost[i, j] for i in set_F for j in set_D)  # Minimum total distance

        #
        # plt.figure(figsize=(5, 5))
        # plt.title(self.name + ' Problem(Demand=' + str(len(self.demand_nodes)) + ',Facility=' + str(
        #     len(self.facility_nodes)) + ')')
        # plt.scatter(*zip(*self.facility_nodes), c='#ff69E1', marker='*', s=20, label='Facility')
        # plt.scatter(*zip(*self.demand_nodes), c='Blue', marker='o', s=20, label='Demand',zorder=0)
        # plt.grid(False)
        # plt.legend(loc='best', fontsize=10)
        # plt.show()
        #
        selected_facility, unselected_facility, assigned = LRP_capModel.show_result(self, prob)
        # LRP_capModel.draw_plot(self, selected_facility, unselected_facility, assigned)

        return selected_facility, unselected_facility, assigned




class LRP_costModel(LocateLRP):
    def __init__(self, facility_nodes, demand_nodes, solver,fa_cap, de_demand, fa_cost):
        super().__init__(facility_nodes, demand_nodes, solver)
        self.name = None
        self.fa_cap = fa_cap
        self.de_demand = de_demand
        self.fa_cost = fa_cost


    def show_result(self, prob):
        global selected, selected_facility, unselected, unselected_facility, assigned
        prob.solve(self.solver)
        print("Status:", LpStatus[prob.status])
        if LpStatus[prob.status] == "Optimal":
            selected, selected_facility, unselected, unselected_facility, assigned = [], [], [],[],[]
            for i in range(len(self.facility_nodes)):
                if self.y[i].varValue == 1:
                    selected.append(i)
                    selected_facility.append(self.facility_nodes[i])
                    for j in range(len(self.demand_nodes)):
                        if self.x[i, j].varValue == 1:
                            assigned.append((i, j))
                else:
                    unselected.append(i)
                    unselected_facility.append(self.facility_nodes[i])

        print("Selected facilities =", selected)
        print("Unselected facilities =", unselected)
        print("Assigned relationships = ", assigned)
        print("Minimum total distance = ", value(prob.objective))

        return selected_facility, unselected_facility,assigned

    def draw_plot(self, selected_facility, unselected_facility, assigned):
        plt.figure(figsize=(5, 5))
        plt.title(self.name + ' Problem(Demand=' + str(len(self.demand_nodes)) + ',Facility=' + str(len(self.facility_nodes)) + ')')
        plt.scatter(*zip(*selected_facility), c='Red', marker='*', s=20, label='Served Facility')
        if len(unselected_facility) != 0:
            plt.scatter(*zip(*unselected_facility), c='#ff69E1', marker='*', s=30, label='Unserved Facility')
        plt.scatter(*zip(*self.demand_nodes), c='Blue', marker='o', s=20, label='Demand')
        for i in range(len(assigned)):
            pts = [self.facility_nodes[assigned[i][0]], self.demand_nodes[assigned[i][1]]]
            plt.plot(*zip(*pts), c='Orange', linewidth=0.5)
        plt.grid(False)
        plt.legend(loc='best', fontsize=10)
        plt.show()


class LRP_cost(LRP_costModel):

    def __init__(self, facility_nodes, demand_nodes, solver, fa_cap, de_demand, fa_cost):
        super().__init__(facility_nodes, demand_nodes, solver, fa_cap, de_demand, fa_cost)
        self.x = None
        self.y = None
        self.name = 'LRP_cost'

    def prob_solve(self):
        global selected, selected_facility, unselected, unselected_facility, assigned
        # Compute euclidean distances between each warehouse and demand point.
        num_fa = len(self.facility_nodes)
        num_de = len(self.demand_nodes)
        cost = np.zeros((num_fa, num_de))
        for i in range(num_fa):
            fa = self.facility_nodes[i]
            for j in range(num_de):
                de = self.demand_nodes[j]
                dx = fa[0] - de[0]
                dy = fa[1] - de[1]
                cost[i][j] = sqrt(dx * dx + dy * dy)
        # use cost[i,j] to get the num
        # Create a new model
        prob = LpProblem("Location_Routing_Problem", LpMinimize)

        # Create variables
        set_F = range(0, num_fa)
        set_D = range(0, num_de)
        # Variables
        x = {(i, j):
                 LpVariable(cat=LpBinary, name="x_{0}_{1}".format(i, j))
             for i in set_F for j in set_D}

        y = {(i):
                 LpVariable(cat=LpBinary, name="y_{0}".format(i))
             for i in set_F}
        self.y = y
        self.x = x

        # Add constraints
        for i in set_F:
            prob += (lpSum([x[i, j] * self.de_demand[j] for j in set_D]) <= self.fa_cap[i][1] * y[i])

        for i in set_F:
            prob += (lpSum([x[i, j] * self.de_demand[j] for j in set_D]) >= self.fa_cap[i][0] * y[i])

        constraints_eq = {j: prob.addConstraint(
            LpConstraint(
                e=lpSum(x[i, j] for i in set_F),
                sense=LpConstraintEQ,
                rhs=1,
                name="constraint_eq_{0}".format(j)
            )
        ) for j in set_D}

        # Set objective
        prob += lpSum(x[i, j] * cost[i, j] for i in set_F for j in set_D)+lpSum(y[i] * self.fa_cost[i] for i in set_F) # Minimum total distance

        #
        # plt.figure(figsize=(5, 5))
        # plt.title(self.name + ' Problem(Demand=' + str(len(self.demand_nodes)) + ',Facility=' + str(
        #     len(self.facility_nodes)) + ')')
        # plt.scatter(*zip(*self.facility_nodes), c='#ff69E1', marker='*', s=20, label='Facility')
        # plt.scatter(*zip(*self.demand_nodes), c='Blue', marker='o', s=20, label='Demand')
        # plt.grid(False)
        # plt.legend(loc='best', fontsize=10)
        # plt.show()
        #
        selected_facility, unselected_facility, assigned = LRP_costModel.show_result(self, prob)
        # LRP_costModel.draw_plot(self, selected_facility, unselected_facility, assigned)
        return selected_facility, unselected_facility, assigned