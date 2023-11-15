import numpy as np
from mysqp import mysqp
from report import report

# Define your objective and constraint functions here.
def objective(x):
    # Replace with your objective function.
    return x[0]**2 + (x[1] - 3)**2

def objectiveg(x):
    # Replace with the gradient of your objective function.
    return np.array([2 * x[0], 2 * (x[1] - 3)])

def constraint(x):
    # Replace with your constraint function.
    return np.array([x[1]**2 - 2 * x[0], (x[1] - 1)**2 + 5 * x[0] - 15])

def constraintg(x):
    # Replace with the gradient of your constraint function.
    # return np.array([[-2, 2 * x[1]], [5, 2 * x[1] - 2]])
    # x[1] = [2, 0]
    C = np.array([[-2, 2 * x[1]], [5, 2 * x[1] - 2]])
    return np.concatenate(C)

# Optimization settings
opt = {
    'alg': 'matlabqp',  # 'myqp' or 'matlabqp'
    'linesearch': True,  # False or True
    'eps': 1e-3,
}

# Initial guess
x0 = np.array([1, 1])

# Feasibility check for the initial point.
if np.max(constraint(x0) > 0):
    print("Infeasible initial point! You need to start from a feasible one.")
else:
    # Run your implementation of SQP algorithm. See mysqp.py
    solution = mysqp(objective, objectiveg, constraint, constraintg, x0, opt)

    # Report
    report(solution, objective, constraint)
