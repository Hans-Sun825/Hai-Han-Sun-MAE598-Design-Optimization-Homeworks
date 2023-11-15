import matplotlib.pyplot as plt
from drawcontour import drawContour
import numpy as np

def report(solution, f, g):
    # Create a contour plot for the objective function
    plt.figure()
    drawContour(f, g)
    
    # Plot the search path
    x = solution['x']
    iter = len(x[0])
    
    x = np.array(x)
    plt.plot(x[0, 0], x[1, 0], '.y', markersize=20)
    
    for i in range(1, iter):
        plt.plot([x[0, i - 1], x[0, i]], [x[1, i - 1], x[1, i]], color='y')
        plt.plot(x[0, i], x[1, i], '.y', markersize=20)
    
    plt.plot(x[0, iter - 1], x[1, iter - 1], '*k', markersize=20)
    
    # Plot the convergence
    F = [f(x[:, i]) for i in range(iter)]
    plt.figure()
    plt.plot(range(1, iter + 1), [np.log(F[i] - F[-1] + np.finfo(float).eps) for i in range(iter)], 'k', linewidth=3)
    plt.show()
