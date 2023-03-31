from location import *
from location import Locate


class LocationSetCoveringModel(Locate):
    def __init__(self, demand, num_points, num_facilities, cover, solver):
        super().__init__(num_points, cover, solver)
        self.demand = demand
        self.num_facilities = num_facilities
        self.x = None
        self.name = 'CoveringModel'

    def prob_solve(self):
        # Build a problem model
        prob = LpProblem("Covering_Model_Problem", LpMinimize)
        # Create variables
        Zones = list(range(self.num_facilities))
        x = LpVariable.dicts("Select", Zones, cat="Binary")  # X
        self.x = x
        # Set objective
        prob += lpSum(x[i] for i in range(self.num_facilities))  # Minimum total cost
        # Add constraints
        for i in range(self.num_points):
            prob += (lpSum([x[j] * self.cover[i][j] for j in range(self.num_facilities)]) >= 1)
        prob.solve(self.solver)

        # Print solution status
        print("Status:", LpStatus[prob.status])

        # Print results
        if LpStatus[prob.status] == "Optimal":
            selected = []
            for i in range(self.num_facilities):
                if x[i].varValue == 1:
                    selected.append(i)
            obj = value(prob.objective)
            print("Selected points =", selected)
            print("Minimum cost =", obj)

        return selected, obj


class MaximumCoveringModel(Locate):
    def __init__(self, num_located, num_people, num_points, cover, solver):
        super().__init__(num_points, cover, solver)
        self.num_located = num_located
        self.num_people = num_people
        self.name = 'MaximumCovering'

    def prob_solve(self):
        # Create a new model
        prob = LpProblem("Maximum_Covering_Model", LpMaximize)

        # Create variables
        Zones = list(range(self.num_points))
        x = LpVariable.dicts("Select", Zones, cat="Binary")  # X
        z = LpVariable.dicts("Serve", Zones, cat="Binary")  # Z

        # Set objective
        prob += lpSum(z[i] * self.num_people[i] for i in range(self.num_points))  # Maximum number of people served

        # Add constraints
        prob += (lpSum([x[j] for j in range(self.num_points)]) == self.num_located)
        for i in range(self.num_points):
            prob += (lpSum([x[j] * self.cover[i][j] for j in range(self.num_points)]) >= z[i])
        prob.solve(self.solver)
        print("Status:", LpStatus[prob.status])

        # Print results
        if LpStatus[prob.status] == "Optimal":
            selected = []
            served = []
            for i in range(self.num_points):
                if x[i].varValue == 1:
                    selected.append(i)
                if z[i].varValue == 1:
                    served.append(i)

        print("Selected position = ", selected)
        print("Served position = ", served)
        print("Max served number = ", value(prob.objective))
        return selected, served, value(prob.objective)


class MaximumExpectedCoverageLocation(Locate):
    def __init__(self, demand, unprob_rate, num_located, num_points, cover, solver):
        super().__init__(num_points, cover, solver)
        self.demand = demand
        self.unprob_rate = unprob_rate
        self.num_located = num_located

    def cal_x_sum(self, x_list):
        """
        求决策变量之和

        :param x_list: 决策变量列表
        :returns: 决策变量之和
        """
        for i in range(0, self.num_points):
            if i == 0:
                ret = x_list[0]
            else:
                ret = ret + x_list[i]
        return ret

    def cal_demand(self, id):
        """
        求需求量对应的节点编号

        :param id: 当前节点编号
        :returns: 能满足 id节点需求的节点列表
        """
        arr = self.cover[id]
        node_list = []
        index = 0
        for i in arr:
            if i == 1:
                node_list.append(index)
            index += 1
        return node_list

    def prob_solve(self):
        prob = pulp.LpProblem("MEXCLP", LpMaximize)
        x_list = []  # X_i
        y_list = []  # y_j_k
        sum_list = []  # Z
        # 定义决策变量
        for i in range(0, self.num_points):
            x_name = 'X_' + str(i)
            x_list.append(pulp.LpVariable(x_name, 0, self.num_located, LpBinary))
            y_m_list = []
            for j in range(1, self.num_located + 1):
                y_name = 'Y_' + str(j) + '_' + str(i)
                y_m_list.append(pulp.LpVariable(y_name, 0, 1, LpBinary))
            y_list.append(y_m_list)

        # 定义函数
        for i in range(0, self.num_points):
            for j in range(1, self.num_located + 1):
                sum_list.append(
                     (1 - self.unprob_rate) * self.unprob_rate ** (j - 1) * self.demand[i] * y_list[i][j - 1])

        # 设置目标函数
        prob += pulp.lpSum(sum_list)
        # 约束 2
        ret_x = MaximumExpectedCoverageLocation.cal_x_sum(self, x_list)
        prob += ret_x <= self.num_located

        for i in range(0, self.num_points):
            x_sum, y_sum = x_list[0], y_list[0][0]
            for j in range(0, self.num_located):
                if j == 0:
                    y_sum = y_list[i][j]
                else:
                    y_sum = y_sum + y_list[i][j]
            d_list = MaximumExpectedCoverageLocation.cal_demand(self, i)
            for k in range(0, len(d_list)):
                if k == 0:
                    x_sum = x_list[d_list[k]]
                else:
                    x_sum = x_sum + x_list[d_list[k]]
            prob += y_sum <= x_sum
        prob.solve(self.solver)  # solver

        print("Status:", LpStatus[prob.status])

        # Print results
        if LpStatus[prob.status] == "Optimal":
            selected = []
            served = []
            for var in prob.variables():
                if var.name.split("_")[0] == "X":
                    for i in range(1, self.num_located + 1):
                        if var.varValue == i:
                            if i == 1:
                                selected.append(int(var.name.split("_")[1]))
                            else:
                                selected = [var.name.split("_")[1] for j in range(i)]
        func = round(pulp.value(prob.objective), 3)

        print("Selected position = ", selected)
        print("Objective function value = ", func)
        print("Unreliability rate = ", self.unprob_rate)

        return selected, served, func


class BackupCoveringModel(Locate):
    def __init__(self, num_facilities, setup_cost, num_points, cover, solver):
        super().__init__(num_points, cover, solver)
        self.num_facilities = num_facilities
        self.setup_cost = setup_cost

    def prob_solve(self):
        # Build a problem model
        global selected, served_once, served_twice
        prob = LpProblem("Backup_Coverage_Location_Problem", LpMaximize)

        # Create variables
        Zones = list(range(self.num_points))
        x = LpVariable.dicts("Select", Zones, cat="Binary")
        y = LpVariable.dicts("Serve", Zones, cat="Binary")
        u = LpVariable.dicts("Twice", Zones, cat="Binary")
        Z1 = lpSum(y[i] * self.setup_cost[i] for i in range(self.num_points))
        Z2 = lpSum(u[i] * self.setup_cost[i] for i in range(self.num_points))
        # Set objective
        prob += Z1 + Z2
        for i in range(self.num_points):
            prob += lpSum([x[j]*self.cover[i][j] for j in range(self.num_points)]) - y[i] - u[i] >= 0
            prob += u[i] - y[i] <= 0

        prob += lpSum(x[i] for i in range(self.num_points)) == self.num_facilities
        prob.solve(self.solver)
        print("Status:", LpStatus[prob.status])
        if LpStatus[prob.status] == "Optimal":
            selected = []
            served_once = []
            served_twice = []
            for j in range(self.num_points):
                if x[j].varValue == 1:
                    selected.append(j)
            for i in range(self.num_points):
                if y[i].varValue == 1:
                    served_once.append(i)
                if u[i].varValue == 1:
                    served_twice.append(i)

            print("Selected position = ", selected)
            print("Served once = ", served_once)
            print("Served twice = ", served_twice)
            print("Max served number = ", value(prob.objective))
        return selected, served_once, served_twice, value(prob.objective)