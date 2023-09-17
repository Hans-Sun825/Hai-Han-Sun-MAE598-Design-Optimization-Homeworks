"""
2.
(a) Find the point in the plane x_1 + 2*x_2 + 3*x_3 = 1 in R^3 that is nearest to the point (-1,0,1)^T. 
Is this a convex problem? 
(b) Implement the gradient descent and Newton's algorithm for solving this problem. Attach a python codes which include: 
(1) The initial points tested; (2) corresponding solutions; (3) A log-linear convergence plot
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

### Define the objective function and its gradient
# x[0] = x_1, x[1] = x_2, x[2] = x_3
def objective(x):
    return ((x[0])**2 + x[1]**2 + (x[2] - 1)**2)**0.5

def gradient(x):
    return np.array([2 * (x[0] + 1), 2 * x[1], 2 * (x[2] - 1)])

# Define the constraint
def constraint(x):
    return x[0] + 2 * x[1] + 3 * x[2] - 1

# Define the gradient constraint
def constraint_gradient(x):
    return np.array([1, 2, 3])

# Define the necessary parameters
X0 = np.array([10.0, 10.0, -10.0])  # X0 : Initial point
alpha = 0.1                         # alpha : learning rate
num_iterations = 50                 # num_iterations : k
epsilon = 0.0001                    # Stopping criteria is abs(f(x)) < epsilon.

# Define the gradient descent
def gradient_descent(alpha, num_iterations, epsilon):
    x = np.array([0.0, 0.0, 0.0])  # Initial point
    storage = []  # To store objective values for convergence plot

    for i in range(num_iterations):
        gradient_x = gradient(x)
        x = x - alpha * gradient_x
        storage.append(objective(x))
        if len(gradient_x)**2 < epsilon:
            break
    return x, storage

# Define the Newton's algorithm
def newton_method(num_iterations):
    x = np.array([0.0, 0.0, 0.0])  # Initial point
    storage = []  # To store objective values for convergence plot

    for i in range(num_iterations):
        gradient_x = gradient(x)
        constraint_gradient_x = constraint_gradient(x)
        # Update x using Newton's method
        x = x - gradient_x.T * gradient_x
        storage.append(objective(x))

    return x, storage

### Import initial guesses and show the results

# Gradient Descent
for initial_point in X0:
    x_gd, storage_gd = gradient_descent(alpha, num_iterations, epsilon)
    print(f"Gradient Descent: Initial Point = {initial_point}, Solution = {x_gd}")
# Newton's Algorithm
for initial_point in X0:
    x_newton, stroage_newton = newton_method(num_iterations)
    print(f"Newton's Method: Initial Point = {initial_point}, Solution = {x_newton}")

# Plot convergence
plt.figure(figsize=(12, 6))
plt.plot(range(num_iterations), storage_gd, label="Gradient Descent", linestyle='--')
plt.plot(range(num_iterations), stroage_newton, label="Newton's Method", linestyle='-')
plt.xlabel("Iterations")
plt.ylabel("Objective Value")
plt.yscale('log')
plt.legend()
plt.title("Convergence Plot")
plt.show()