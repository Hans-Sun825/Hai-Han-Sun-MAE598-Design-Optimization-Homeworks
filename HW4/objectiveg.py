import numpy as np

def objectiveg(x):
    # Calculate the gradient of the objective (row vector)
    # df = [df/dx1, df/dx2, ..., df/xn]
    df = np.array([2*x[0], 2*x[1] - 6])
    return df
