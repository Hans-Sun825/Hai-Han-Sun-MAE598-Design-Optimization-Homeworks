import numpy as np
from scipy.linalg import solve

def solveqp(x, W, df, g, dg):
    # Compute c in the QP problem formulation
    c = df(x)

    # Compute A in the QP problem formulation
    A0 = dg(x)

    # Compute b in the QP problem formulation
    b0 = -g(x)

    # Initialize variables for the active-set strategy
    stop = 0  # Start with stop = 0
    A = np.array([])  # A for empty working-set
    b = np.array([])  # b for empty working-set
    active = np.array([], dtype=int)  # Indices of the constraints in the working-set

    while not stop:
        # Initialize all mu as zero and update the mu in the working set
        mu0 = np.zeros(g(x).shape)

        # Extract A corresponding to the working-set
        A = A0[active, :]
        # Extract b corresponding to the working-set
        b = b0[active]

        # Solve the QP problem given A and b
        s, mu = solve_activeset(x, W, c, A, b)
        # Round mu to prevent numerical errors
        mu = np.round(mu * 1e12) / 1e12

        # Update mu values for the working-set using the solved mu values
        mu0[active] = mu

        # Calculate the constraint values using the solved s values
        gcheck = A0 @ s - b0
        # Round constraint values to prevent numerical errors
        gcheck = np.round(gcheck * 1e12) / 1e12

        # Variable to check if all mu values make sense
        mucheck = 0  # Initially set to 0

        # Indices of the constraints to be added to the working set
        Iadd = np.array([], dtype=int)
        # Indices of the constraints to be removed from the working set
        Iremove = np.array([], dtype=int)

        # Check mu values and set mucheck to 1 when they make sense
        if len(mu) == 0:
            # When there are no mu values in the set
            mucheck = 1  # OK
        elif np.min(mu) > 0:
            # When all mu values in the set are positive
            mucheck = 1  # OK
        else:
            # When some of the mu are negative
            # Find the most negative mu and remove it from the active set
            Iremove = np.argmin(mu)

        # Check if constraints are satisfied
        if np.max(gcheck) <= 0:
            # If all constraints are satisfied
            if mucheck == 1:
                # If all mu values are OK, terminate by setting stop = 1
                stop = 1
        else:
            # If some constraints are violated
            # Find the most violated one and add it to the working set
            Iadd = np.argmax(gcheck)

        # Remove the index Iremove from the working-set
        active = np.delete(active, Iremove)
        # Add the index Iadd to the working-set
        active = np.append(active, Iadd)

        # Make sure there are no duplications in the working-set
        active = np.unique(active)

    return s, mu0

def solve_activeset(x, W, c, A, b):
    # Given an active set, solve QP
    
    # Create the linear set of equations given in equation (7.79)
    n = len(x)
    m = len(b)
    M = np.zeros((n + m, n + m))
    # [:n] : delete values behind number n
    # [n:] : keep values behind number n
    M[:n, :n] = W
    M[:n, n:] = A.T
    M[n:, :n] = A
    U = np.concatenate([-c, b])

    sol = np.linalg.solve(M, U)

    s = sol[:n]  # Extract s from the solution
    mu = sol[n:]  # Extract mu from the solution

    return s, mu
# Add the report and drawContour functions from your previous code to visualize the results.

# Example usage:
# solution = mysqp(f, df, g, dg, x0, opt)
# report(solution, f, g)
