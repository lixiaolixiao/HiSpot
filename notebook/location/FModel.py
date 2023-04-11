from pulp import *
from location import Locate
from location import Flow
import pulp

class MaximumFlowInterceptionModel(Flow):
    def __init__(self, num_path, num_vector, num_choice, path_vector, path_flow, solver):
        super().__init__(num_path, num_vector, path_vector, solver)
        self.num_choice = num_choice
        self.path_flow = path_flow
        self.name = 'Maximization of the Intercepted Flow'
        self.xp = None
        self.yi = None

    def prob_solve(self):
        # Build a problem model
        prob = LpProblem("Maximization of the Intercepted Flow", LpMaximize)
        xp = {}
        yi = {}
        # Create variables
        for i in self.num_path:
            name = 'Select_Path' + str(i)
            xp[i] = pulp.LpVariable(name, 0, 1, pulp.LpBinary)

        for i in self.num_vector:
            name = 'Select_Vector' + str(i)
            yi[i] = pulp.LpVariable(name, 0, 1, pulp.LpBinary)
        # zones_p = list(range(self.num_path))
        # zones_v = list(range(self.num_vector))
        # xp = LpVariable.dicts("Select_Path", zones_p, cat="Binary")  # xp
        # yi = LpVariable.dicts("Select_Vector", zones_v, cat="Binary")  # yi
        self.xp = xp
        self.yi = yi

        # Set objective
        prob += pulp.lpSum(self.path_flow[i] * self.xp[i] for i in self.num_path)

        # Add constraints
        prob += pulp.lpSum(yi[i] for i in self.num_vector) == self.num_choice

        for k in self.num_path:
            prob += pulp.lpSum(yi[i] for i in self.path_vector[k]) >= self.xp[k]

        # solve the problem
        prob.solve(self.solver)

        # Print solution status
        print("Status:", LpStatus[prob.status])
        # Print results
        if LpStatus[prob.status] == "Optimal":
            selected_path = []
            selected_vector = []
            for i in self.num_path:
                if xp[i].varValue == 1:
                    selected_path.append(i)
            for i in self.num_vector:
                if yi[i].varValue == 1:
                    selected_vector.append(i)

            print("Selected paths =", selected_path)
            print("Selected points =", selected_vector)
            print("Maximum flow =", value(prob.objective))
        return selected_path, selected_vector

class MinimumFlowInterceptionModel(Flow):
    def __init__(self, num_path, num_vector, path_vector, path_flow, intercept_e, solver):
        super().__init__(num_path, num_vector, path_vector, solver)
        self.path_flow = path_flow
        self.intercept_e = intercept_e
        self.name = 'Minimization of the Intercepted Flow'
        self.xp = None
        self.yi = None

    def prob_solve(self):
        # Build a problem model
        prob = LpProblem("Minimization of the Intercepted Flow", LpMinimize)

        # Create variables
        xp = {}
        yi = {}
        # Create variables
        for i in self.num_path:
            name = 'Select_Path' + str(i)
            xp[i] = pulp.LpVariable(name, 0, 1, pulp.LpBinary)

        for i in self.num_vector:
            name = 'Select_Vector' + str(i)
            yi[i] = pulp.LpVariable(name, 0, 1, pulp.LpBinary)

        # Set objective
        prob += pulp.lpSum(yi[i] for i in self.num_vector)

        # Add constraints
        for p in self.num_path:
            prob += pulp.lpSum(yi[i] for i in self.path_vector[p]) >= xp[p]

        prob += pulp.lpSum(self.path_flow[p] * xp[p] for p in self.num_path) >= self.intercept_e

        # solve the problem
        prob.solve(self.solver)

        # Print solution status
        print("Status:", LpStatus[prob.status])

        # Print results
        if LpStatus[prob.status] == "Optimal":
            selected_path = []
            selected_vector = []
            for i in self.num_path:
                if xp[i].varValue == 1:
                    selected_path.append(i)
            for i in self.num_vector:
                if yi[i].varValue == 1:
                    selected_vector.append(i)

            print("Selected paths =", selected_path)
            print("Selected points =", selected_vector)
            print("Minimum flow =", value(prob.objective))

        return selected_path,selected_vector


class MaximizationAchievableGain(Flow):
    def __init__(self, num_path, num_vector, num_choice, path_vector, vector_gain, solver):
        super().__init__(num_path, num_vector, path_vector, solver)
        self.num_choice = num_choice
        self.vector_gain = vector_gain
        self.name = 'Maximization of the Intercepted Flow'
        self.xpi = None
        self.yi = None

    def prob_solve(self):
        # Build a problem model
        prob = LpProblem("Maximization of the Achievable Gain", LpMaximize)

        # Create variables
        xpi = {}
        yi={}
        for i in self.num_path:
            for j in self.path_vector[i]:
                name = 'Select_Path' + str(i) + '_' + str(j)
                xpi[i, j] = pulp.LpVariable(name, 0, 1, LpBinary)
        for i in self.num_vector:
            name = 'Y' + str(i)
            yi[i] = pulp.LpVariable(name, 0, 1, pulp.LpBinary)
        self.xpi = xpi
        self.yi = yi

        # Set objective
        prob += pulp.lpSum(self.vector_gain[p, i%2] * xpi[p, i]
                           for p in self.num_path for i in self.path_vector[p])

        # Add constraints
        prob += pulp.lpSum(yi[i] for i in self.num_vector) == self.num_choice

        for p in self.num_path:
            for k in self.path_vector[p]:
                prob += pulp.lpSum(yi[i] for i in self.path_vector[p]) >= xpi[p, k]

        for p in self.num_path:
            prob += pulp.lpSum(xpi[p, i] for i in self.path_vector[p]) <= 1

        # solve the problem
        prob.solve(self.solver)

        # Print solution status
        print("Status:", LpStatus[prob.status])

        # Print results
        if LpStatus[prob.status] == "Optimal":
            selected_path = []
            selected_vector = []
            for i in self.num_path:
                for j in self.path_vector[i]:
                    if xpi[i, j].varValue == 1:
                        selected_path.append(i)
                        break
            for i in self.num_vector:
                if yi[i].varValue == 1:
                    selected_vector.append(i)

            print("Selected paths =", selected_path)
            print("Selected points =", selected_vector)
            print("Maximum flow =", value(prob.objective))

        return selected_path, selected_vector

class MinimumFacilityMaximumGain(Flow):
    def __init__(self, num_path, num_vector, path_vector, vector_gain,flow_gain, solver):
        super().__init__(num_path, num_vector, path_vector, solver)
        self.vector_gain=vector_gain
        self.flow_gain = flow_gain
        self.name = 'Minimization of the Number of Facilities for Gain Maximization'
        self.xpi = None
        self.yi = None

    def prob_solve(self):
        # Build a problem model
        prob = LpProblem('Minimization of the Number of Facilities for Gain Maximization', LpMinimize)

        # Create variables
        xpi = {}
        yi = {}
        for i in self.num_path:
            for j in self.path_vector[i]:
                name = 'Select_Path' + str(i) + '_' + str(j)
                xpi[i, j] = pulp.LpVariable(name, 0, 1, LpBinary)
        for i in self.num_vector:
            name = 'Y' + str(i)
            yi[i] = pulp.LpVariable(name, 0, 1, pulp.LpBinary)
        self.xpi = xpi
        self.yi = yi

        # Set objective
        prob += pulp.lpSum(yi[i] for i in self.num_vector)

        # Add constraints
        for p in self.num_path:
            for k in self.path_vector[p]:
                prob += pulp.lpSum(yi[i] for i in self.path_vector[p]) >= xpi[p, k]

        for p in self.num_path:
            prob += pulp.lpSum(xpi[p, i] for i in self.path_vector[p]) <= 1

        prob += pulp.lpSum(self.vector_gain[p, i%2] * xpi[p, i]
                           for p in self.num_path for i in self.path_vector[p]) >= self.flow_gain

        # solve the problem
        prob.solve(self.solver)

        # Print solution status
        print("Status:", LpStatus[prob.status])

        # Print results
        selected_path = []
        selected_vector = []
        if LpStatus[prob.status] == "Optimal":

            for i in self.num_path:
                for j in self.path_vector[i]:
                    if xpi[i, j].varValue == 1:
                        selected_path.append(i)
                        break
            for i in self.num_vector:
                if yi[i].varValue == 1:
                    selected_vector.append(i)

            print("Selected paths =", selected_path)
            print("Selected points =", selected_vector)
            print("Minimization of the Number of Facilities for Gain Maximization = ", value(prob.objective))

        return selected_path, selected_vector
