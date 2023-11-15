import numpy as np
from scipy.optimize import minimize, LinearConstraint, lsq_linear
import pandas as pd
#from lineSearch import lineSearch
#from solveqp import solveqp

def mysqp(f, df, g, dg, x0, opt):
    # Set initial conditions
    x = x0  # Set current solution to the initial guess

    # Initialize a list to record search process
    solution = {'x': [x.copy()]}  # save current solution to solution['x']

    # Initialization of the Hessian matrix
    W = np.eye(len(x))  # Start with an identity Hessian matrix

    # Initialization of the Lagrange multipliers
    mu_old = np.zeros(len(g(x)))  # Start with zero Lagrange multiplier estimates

    # Initialization of the weights in the merit function
    w = np.zeros(len(g(x)))  # Start with zero weights

    # Reshape dg(x) to a 2x2 matrix
    dg_matrix = np.reshape(dg(x), (2, 2))

    # Set the termination criterion
    gnorm = np.linalg.norm(df(x) + np.matmul(mu_old, dg_matrix))  # norm of Lagrangian gradient

    while gnorm > opt['eps']:  # if not terminated
        # Implement QP problem and solve
        if opt['alg'] == 'myqp':
            # Solve the QP subproblem to find s and mu (using your own method)
            s, mu_new = solveqp(x, W, df, g, dg)
        else:
            # Solve the QP subproblem to find s and mu (using scipy's solver)
            #objective = lambda s: 0.5 * np.dot(np.transpose(s), np.dot(W, s)) + np.dot(np.transpose(df(x)), s)
            # x0=np.ones(len(x)), constraints=[cons], options={'disp': False})
            # [x,fval,exitflag,output,lambda] = quadprog(H,f,A,b,Aeq,beq,lb,ub,x0,options)
            # minimizes 1/2*x'*H*x + f'*x subject to the restrictions A*x ≤ b.
            # minimizes 0.5*s'*W*s + [df(x)]'*s s.t. [dg(x)]'*s ≤ -g(x)

            # Solve the QP subproblem to find s and mu (using scipy's solver)
            cons = LinearConstraint(A = np.transpose(-dg_matrix), lb=-np.inf, ub=-g(x))
            # lambda s: 0.5 * np.dot(s, np.dot(W, s)) + np.dot(df(x), s)
            result = minimize(lambda s: 0.5 * np.dot(np.transpose(s), np.dot(W, s)) + np.dot(np.transpose(df(x)), s), 
                              x0=np.ones(len(x)), constraints=[cons], options={'disp': False})
            
            s = result.x
            #mu_new = result.get('dual', np.zeros(len(g(x))))[:len(g(x))]
            mu_new = result.get('dual', None)
            if mu_new is not None:
                mu_new = mu_new[:len(g(x))]
            else:
                mu_new = np.zeros(len(g(x)))

        # opt['linesearch'] switches line search on or off.
        # You can first set the variable "a" to different constant values and see how it
        # affects the convergence.
        if opt['linesearch']:
            a, w = lineSearch(f, df, g, dg, x, s, mu_old, w)
        else:
            a = 0.1

        # Update the current solution using the step
        dx = a * s  # Step for x
        x = x + dx  # Update x using the step
        
        # Reshape s to ensure it's a 1D array
        s = s.reshape((-1,))

        # Update Hessian using BFGS. Use equations (7.36), (7.73) and (7.74)
        # Compute y_k
        dg_minus_matrix = np.reshape(dg(x - dx), (2, 2))
        y_k = df(x) + np.dot(mu_new, dg_matrix) - df(x - dx) - np.dot(mu_new, dg_minus_matrix)
        y_k = y_k.reshape((-1, 1))
        # Compute theta
        # if dx'*y_k >= 0.2*dx'*W*dx
        if np.dot(np.transpose(dx), y_k) >= 0.2 * np.dot(np.transpose(dx), np.dot(W, dx)):
            theta = 1
        else:
            #theta = (0.8*dx'*W*dx)/(dx'*W*dx-dx'*y_k);
            theta = (0.8 * np.dot(np.transpose(dx), np.dot(W, dx))) / (np.dot(np.transpose(dx), np.dot(W, dx)) - np.dot(np.transpose(dx), y_k))
        # Compute dg_k
        dg_k = theta * y_k + (1 - theta) * np.dot(W, dx)
        # Update Hessian using BFGS
        dx_trans = np.transpose(dx)  # Reshape dx to a column vector
        dg_k_trans = np.transpose(dg_k) # dg_k transpose

        # Update Hessian using BFGS
        # W = W + (dg_k*dg_k')/(dg_k'*dx) - ((W*dx)*(W*dx)')/(dx'*W*dx);
        W_dx = np.dot(W, dx)
        W_dx_trans = np.transpose(W_dx)
        W = W + np.dot(dg_k, dg_k_trans) / np.dot(dg_k_trans, dx) - np.dot(W_dx, W_dx_trans) / np.dot(dx_trans, np.dot(W, dx))

        # Update termination criterion:
        gnorm = np.linalg.norm(df(x) + np.matmul(mu_new, dg_matrix))  # norm of Lagrangian gradient
        mu_old = mu_new

        # save current solution to solution['x']
        solution['x'].append(x.copy())

    return solution

def solveqp(x, W, df, g, dg):
    # Implement your own method to solve the QP subproblem
    # This is a placeholder, replace it with your own implementation
    s = np.zeros(len(x))
    mu = np.zeros(len(g(x)))
    return s, mu

def lineSearch(f, df, g, dg, x, s, mu, w):
    # Implement your own line search method
    # This is a placeholder, replace it with your own implementation
    a = 0.1
    return a, w

# Example usage:
# opt = {'alg': 'myqp', 'linesearch': True, 'eps': 1e-6}
# x0 = np.array([1.0, 1.0])  # Initial guess
# result = mysqp(myfunc, myfunc_derivative, myconstraints, myconstraints_derivative, x0, opt)
# print(result)
