import numpy as np
from scipy.optimize import newton

# Global max iterations and tolerance setter (adjustable)
maxiter=100
tolerance=1e-6

def myNR(func, x0, funcderiv=None):
    """
    Compute the Newton-Raphson method (or Secant method).
    If funcderiv is supplied, will perform the Newton-Raphson method.
    Otherwise, will perform the modified Secant method.
    
    Parameters:
    func: function for which the root is sought
    x0: initial guess for the root
    funcderiv: derivative of the function (optional, if None Secant method is used)
    tol: tolerance for convergence (default: 1e-6)
    
    Returns:
    x: root of the function
    """
    # Initialize variables
    x = x0
    iter_count = 0
    
    if funcderiv:  # Newton-Raphson method
        while iter_count < maxiter:
            f_val = func(x)
            f_prime = funcderiv(x)
            if f_prime == 0:
                raise ValueError("Derivative is zero. No solution found.")
            
            # Compute next approximation
            x_new = x - f_val / f_prime
            relative_error = abs((x_new - x) / x_new)
            
            if relative_error < tolerance:
                return x_new, iter_count  # Converged
            
            x = x_new
            iter_count += 1
        raise ValueError("Maximum iterations reached. No solution found.")
    
    else:  # Secant method
        delta = 0.01 * x0  # Small step for modified Secant method
        x_prev = x0 - delta
        
        while iter_count < maxiter:
            f_x = func(x)
            f_x_prev = func(x_prev)
            
            # Compute approximate derivative
            approx_deriv = (f_x - f_x_prev) / (x - x_prev)
            if approx_deriv == 0:
                raise ValueError("Approximate derivative is zero. No solution found.")
            
            # Compute next approximation
            x_new = x - f_x / approx_deriv
            relative_error = abs((x_new - x) / x_new)
            
            if relative_error < tolerance:
                return x_new, iter_count  # Converged
            
            x_prev = x
            x = x_new
            iter_count += 1
        raise ValueError("Maximum iterations reached. No solution found.")

# Function for task 3: f(x) = x^5 - 11x^4 + 43x^3 - 73x^2 + 56x - 16
def test_func(x):
    return x**5 - 11*x**4 + 43*x**3 - 73*x**2 + 56*x - 16

# Derivative of the function: f'(x) = 5x^4 - 44x^3 + 129x^2 - 146x + 56
def test_func_deriv(x):
    return 5*x**4 - 44*x**3 + 129*x**2 - 146*x + 56

# Task 3 - Do the tests specified
if __name__ == '__main__':
    # Test with Newton-Raphson
    root_nr, iterations_nr = myNR(test_func, x0=-2.0, funcderiv=test_func_deriv)
    
    # Test with Secant method (no derivative)
    root_secant, iterations_secant = myNR(test_func, x0=-2.0)

    # Compare with SciPy's Newton method as per Task 4
    root_scipy = newton(test_func, x0=-2.0, fprime=test_func_deriv)

    print("Newton-Raphson root:", root_nr, "Iterations:", iterations_nr)
    print("Secant method root:", root_secant, "Iterations:", iterations_secant)
    print("SciPy Newton root:", root_scipy)
