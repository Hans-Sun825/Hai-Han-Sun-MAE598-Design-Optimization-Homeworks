"""
2.
(a) Find the point in the plane x_1 + 2*x_2 + 3*x_3 = 1 in R^3 that is nearest to the point (-1,0,1)^T. 
Is this a convex problem? 
(b) Implement the gradient descent and Newton's algorithm for solving this problem. Attach a python codes which include: 
(1) The initial points tested; (2) corresponding solutions; (3) A log-linear convergence plot
"""

import numpy as np
import matplotlib.pyplot as plt

# Define the objective function and its gradient
def objective(x):
    return ((x[0] + 1)**2 + x[1]**2 + (x[2] - 1)**2)**0.5

def gradient(x):
    return np.array([(((2*x[1]**2) - 4*x[1] + 6*x[1]*x[2])**(-0.5)) * (2*x[1] - 2 + 3*x[2])],
                    [((10*x[2]**2) - 14*x[2] + 6*x[1]*x[2])**(-0.5) * (10*x[2]- 7 + 6*x[1])])

# Define the constraint
def constraint(x):
    return x[0] + 2 * x[1] + 3 * x[2] - 1

# Define the gradient constraint
def constraint_gradient(x):
    return np.array([1, 2, 3])

# Define the necessary parameters
alpha = 0.1
num_iterations = 50
epsilon = 100
x0 = np.array([0.0, 0.0, 0.0])

# Define the gradient descent
def gradient_descent(alpha, num_iterations):
    x = np.array([0.0, 0.0, 0.0])  # Initial point
    storage = []  # To store objective values for convergence plot

    for i in range(num_iterations):
        x = x - alpha * gradient(x)
        storage.append(objective(x))

    return x, storage

# Define the Newton's algorithm
def newton(f,Df,x0,epsilon,max_iter):
    '''Approximate solution of f(x)=0 by Newton's method.

    Parameters
    ----------
    f : function
        Function for which we are searching for a solution f(x)=0.
    Df : function
        Derivative of f(x).
    x0 : number
        Initial guess for a solution f(x)=0.
    epsilon : number
        Stopping criteria is abs(f(x)) < epsilon.
    max_iter : integer
        Maximum number of iterations of Newton's method.

    Returns
    -------
    xn : number
        Implement Newton's method: compute the linear approximation
        of f(x) at xn and find x intercept by the formula
            x = xn - f(xn)/Df(xn)
        Continue until abs(f(xn)) < epsilon and return xn.
        If Df(xn) == 0, return None. If the number of iterations
        exceeds max_iter, then return None.

    Examples
    --------
    >>> f = lambda x: x**2 - x - 1
    >>> Df = lambda x: 2*x - 1
    >>> newton(f,Df,1,1e-8,10)
    Found solution after 5 iterations.
    1.618033988749989
    '''

    xn = x0
    for n in range(0,max_iter):
        fxn = f(xn)
        if abs(fxn) < epsilon:
            print('Found solution after',n,'iterations.')
            return xn
        Dfxn = Df(xn)
        if Dfxn == 0:
            print('Zero derivative. No solution found.')
            return None
        xn = xn - fxn/Dfxn
    print('Exceeded maximum iterations. No solution found.')
    return None
