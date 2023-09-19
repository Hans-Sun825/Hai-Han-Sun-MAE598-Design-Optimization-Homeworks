"""
2.
(a) Find the point in the plane x_1 + 2*x_2 + 3*x_3 = 1 in R^3 that is nearest to the point (-1,0,1)^T. 
Is this a convex problem? 
(b) Implement the gradient descent and Newton's algorithm for solving this problem. Attach a python codes which include: 
(1) The initial points tested; (2) corresponding solutions; (3) A log-linear convergence plot
"""
##### To check the commit histories, please visit this site:
##### https://github.com/Hans-Sun825/Hai-Han-Sun-MAE598-Design-Optimization-Homeworks 

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import line_search

### Define the objective function and its gradient
# x[0] = x_2, x[1] = x_3
# x_1 = 1 - 2*x_2 - 3*x_3
def objective(x):
    return ((2 - 2*x[0] -3*x[1])**2 + x[0]**2 + (x[1] - 1)**2)**0.5

"""
>>> a = numpy.array([1, 2, 3, 4], dtype=numpy.float64)
>>> a
array([ 1.,  2.,  3.,  4.])
>>> a.astype(numpy.int64)
array([1, 2, 3, 4])
"""

def gradient(x):
    a = np.array([0.5 * ((2 - 2*x[0] - 3*x[1])**2 + x[0]**2 + (x[1]-1)**2)**-0.5 * (2 * (2 - 2*x[0] - 3*x[1]) * (-2) + 2*x[0]),
    0.5 * ((2 - 2*x[0] - 3*x[1])**2 + x[0]**2 + (x[1]-1)**2)**-0.5 * (2 * (2 - 2*x[0] - 3*x[1]) * (-3) + 2 * (x[1]-1))])
    a_int = a.T.astype(int)
    return np.array(a_int)

def hessian(x):
    return([0.805, -5.635], 
           [-5.635, 0.089])

# Define the necessary parameters
X0 = np.array([0.0, 0.0])           # X0 : Initial point
alpha = 1                           # alpha : learning rate
num_iterations = 500                # num_iterations : k
epsilon = 0.0001                    # Stopping criteria is abs(f(x)) < epsilon.

# Define the gradient descent
def gradient_descent(alpha, num_iterations, epsilon):
    x = np.array([0.0, 0.0])        # Initial point
    storage = []                    # To store objective values for convergence plot

    for i in range(num_iterations):
        gradient_x = gradient(x)
        x = x - alpha * gradient_x
        storage.append(objective(x))
        if len(gradient_x)**2 < epsilon:
            break
        else:
            x = x + 1
    return x, storage

# Define the Newton's algorithm
def newton_method(num_iterations,alpha, epsilon):
    x = np.array([0.0, 0.0])        # Initial point
    storage = []                    # To store objective values for convergence plot

    for i in range(num_iterations):
        gradient_x = gradient(x)
        hessian_x = hessian(x)
        # Update x using Newton's method
        x = x - alpha * np.linalg.inv(hessian_x) * gradient_x
        if len(gradient_x) < epsilon:
            break
        else:
            x = x + 1
        storage.append(objective(x))
    return x, storage

### Import initial guesses and show the results

# Gradient Descent
for initial_point in X0:
    x_gd, storage_gd = gradient_descent(alpha, num_iterations, epsilon)
    print(f"Gradient Descent: Initial Point = {initial_point}, Solution = {x_gd}")
# Newton's Algorithm
for initial_point in X0:
    x_newton, stroage_newton = newton_method(num_iterations,alpha, epsilon)
    print(f"Newton's Method: Initial Point = {initial_point}, Solution = {x_newton}")

# Plot convergence
plt.figure(figsize=(12, 6))
plt.plot(range(num_iterations), storage_gd, label="Gradient Descent", linestyle='--')
plt.plot(range(num_iterations), stroage_newton, label="Newton's Method", linestyle='-')
plt.xlabel("Iterations")
plt.ylabel("Objective Value")
plt.yscale("log")
plt.legend()
plt.title("Convergence Plot")
plt.show()