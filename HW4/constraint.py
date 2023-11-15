import numpy as np

def constraint(x):
    # Calculate the constraints (column vector)
    # g = [g1; g2; ... ; gm]
    g1 = x[1]**2 - 2*x[0]
    g2 = (x[1] - 1)**2 + 5*x[0] - 15
    g = np.array([g1, g2])
    return g
