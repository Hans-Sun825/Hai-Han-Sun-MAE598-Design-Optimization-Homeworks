import numpy as np
# The as np portion of the code then tells Python to give NumPy the alias of np. 
# This allows you to use NumPy functions by simply typing np.function_name rather than numpy.function_name.
from scipy.optimize import minimize

# Define the objective function
# x[0] = x1 ; x[1] = x2 ; x[2] = x3 ; x[3] = x4 ; x[4] = x5
def objective_function(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    x5 = x[4]
    return (x1-x2)**2 + (x2+x3-2)**2 + (x4-1)**2 + (x5-1)**2

# Define the constraint functions
def constraint1(x):
    return x[0]+3*x[1]

def constraint2(x):
    return x[2] + x[3] - 2*x[4]

def constraint3(x):
    return x[1] - x[4]

# Initial guesses
initial_guess = [0, 0, 0, 0, 0]

# Define the bounds for variables
b = (-10.0, 10.0)
bounds = [b,b,b,b,b]

# Define the constraints
# equality ('eq') : f(x,y) = c
# inequality ('ineq') : f(x,y) <= c or f(x,y) >= c
constraints = ({'type': 'eq', 'fun': constraint1},
               {'type': 'eq', 'fun': constraint2},
               {'type': 'eq', 'fun': constraint3})

# Solve the optimization problem
result = minimize(objective_function, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

# Print the results
print("Optimal solution:")
print("x_1 =", result.x[0])
print("x_2 =", result.x[1])
print("x_3 =", result.x[2])
print("x_4 =", result.x[3])
print("x_5 =", result.x[4])
print("Optimal value =", result.fun)