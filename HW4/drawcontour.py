import numpy as np
import matplotlib.pyplot as plt

def drawContour(f, g):
    # Define the range of the contour plot
    x = np.arange(-6, 6, 0.1)
    y = np.arange(-6, 6, 0.1)

    # Create a grid of x and y values
    X, Y = np.meshgrid(x, y)

    # Evaluate objective values on the grid
    Zf = np.zeros_like(X)
    Zg1 = np.zeros_like(X)
    Zg2 = np.zeros_like(X)

    for i in range(len(x)):
        for j in range(len(y)):
            point = np.array([x[i], y[j]])
            Zf[j, i] = f(point)
            gall = g(point)
            Zg1[j, i] = gall[0]
            Zg2[j, i] = gall[1]

    # Plot contour
    plt.contourf(x, y, Zf, 100)
    plt.contour(x, y, Zg1, levels=[0], colors='red')
    plt.contour(x, y, Zg2, levels=[0], colors='magenta')
    
    Zg1[Zg1 > 0] = np.nan
    Zg2[Zg2 > 0] = np.nan
    
    plt.contour(x, y, Zg1, levels=10, colors='red')
    plt.contour(x, y, Zg2, levels=10, colors='magenta')
    
    plt.show()

# Example usage:
# Replace `f` and `g` with your actual objective and constraint functions.
# drawContour(f, g)
