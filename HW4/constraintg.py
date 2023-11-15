import numpy as np

def constraintg(x):
    # Calculate the gradient of the constraints
    # dg = [dg1/dx1, dg1/dx2, ... , dg1/dxn;
    #       dg2/dx1, dg2/dx2, ... , dg2/dxn;
    #       ...
    #       dgm/dx1, dgm/dx2, ... , dgm/dxn]
    dg1_dx1 = -2
    dg1_dx2 = 2 * x[1]
    dg2_dx1 = 5
    dg2_dx2 = 2 * x[1] - 2

    dg = np.array([[dg1_dx1, dg1_dx2],
                   [dg2_dx1, dg2_dx2]])
    
    return dg
