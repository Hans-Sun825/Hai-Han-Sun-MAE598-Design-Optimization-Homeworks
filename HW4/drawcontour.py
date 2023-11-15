import numpy as np
import matplotlib.pyplot as plt

def drawContour(f, g):
    # Define the range of the contour plot
    x = np.arange(-6, 6.1, 0.1)
    y = np.arange(-6, 6.1, 0.1)

    # Evaluate objective values on the grid
    Zf = np.zeros((len(y), len(x)))
    Zg1 = Zf.copy()
    Zg2 = Zf.copy()
    
    for i in range(len(x)):
        for j in range(len(y)):
            Zf[j, i] = f([x[i], y[j]])
            gall = g([x[i], y[j]])
            Zg1[j, i] = gall[0]
            Zg2[j, i] = gall[1]

    # Plot contour
    plt.contourf(x, y, Zf, 100)
    plt.contour(x, y, Zg1, levels=[0], colors='r')
    plt.contour(x, y, Zg2, levels=[0], colors='m')
    
    Zg1[Zg1 > 0] = np.nan
    Zg2[Zg2 > 0] = np.nan
    
    plt.contour(x, y, Zg1, levels=10, colors='r')
    plt.contour(x, y, Zg2, levels=10, colors='m')
    
    plt.show()

