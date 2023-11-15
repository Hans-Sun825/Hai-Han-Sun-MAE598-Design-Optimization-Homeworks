import numpy as np

def lineSearch(f, df, g, dg, x, s, mu_old, w_old):
    t = 0.1  # scale factor on current gradient: [0.01, 0.3]
    b = 0.8  # scale factor on backtracking: [0.1, 0.8]
    a = 1.0  # maximum step length
    
    D = s  # direction for x
    
    # Calculate weights in the merit function
    w = np.maximum(np.abs(mu_old), 0.5 * (w_old + np.abs(mu_old)))
    
    # Terminate if line search takes too long
    count = 0
    while count < 100:
        # Calculate phi(alpha) using merit function
        phi_a = f(x + np.array(D) * a) + np.sum(w * np.abs(np.minimum(0, -g(x + np.array(D) * a))))
        
        # Calculate psi(alpha) in the line search using phi(alpha)
        phi0 = f(x) + np.sum(w * np.abs(np.minimum(0, -g(x))))
        dphi0 = np.dot(df(x), D) + np.sum(w * (dg(x) * D) * (g(x) > 0))
        psi_a = phi0 + t * a * dphi0
        
        # Stop if the condition is satisfied
        if np.all(phi_a < psi_a):
            break
        else:
            # Backtracking
            a *= b
            count += 1
    
    return a, w
