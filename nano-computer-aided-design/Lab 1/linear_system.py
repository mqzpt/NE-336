import numpy as np
from scipy import linalg
import time

def solve_system(A, b, method=None, use_scipy=False):
    """
    Solve the linear system [A][x] = [b] using different methods.
    
    A: square matrix (n x n)
    b: column vector (n x 1)
    method: str - the method to use ('inv', 'lu', None)
    use_scipy: boolean - if True, use scipy.linalg.solve, otherwise use numpy.linalg.solve
    
    Returns:
    x: Solution vector
    time_taken: Time taken for the solution
    """
    
    # 1. Test if A is a square matrix
    if A.shape[0] != A.shape[1]:
        print("Matrix [A] is not a square matrix")
        return None
    
    # 2. Test if the dimensions of A and b match
    if A.shape[0] != b.shape[0]:
        print("Matrix dimension mismatch")
        return None
    
    # 3. Test for consistency using matrix rank
    if np.linalg.matrix_rank(A) != A.shape[0]:
        print("Matrix [A] is singular or inconsistent")
        return None
    
    # Measure time taken
    start_time = time.time()
    
    # Solve system using inverse method if specified
    if method == 'inv':
        A_inv = np.linalg.inv(A)
        x = np.dot(A_inv, b)
        
    # Solve system using LU decomposition if specified
    elif method == 'lu':
        P, L, U = linalg.lu(A)
        y = np.linalg.solve(L, b)  # Solve Ly = b
        x = np.linalg.solve(U, y)  # Solve Ux = y
        
    # Use scipy or numpy for solving the system
    else:
        if use_scipy:
            x = linalg.solve(A, b)
        else:
            x = np.linalg.solve(A, b)
    
    time_taken = time.time() - start_time
    return x, time_taken


if __name__ == "__main__":
    # Warmup Tasks
    # First we can define the matrix A and vector b outside of the function
    A = np.array([[-3, 2, -1], [6, -6, 7], [3, -4, 4]])
    b = np.array([-1, -7, -6])
    
    # Task 1: Solve using numpy.linalg.solve
    x_numpy, time_numpy = solve_system(A, b, use_scipy=False)
    print(f"Solution using numpy.linalg.solve: {x_numpy}, Time taken: {time_numpy:.6f} seconds")
    
    # Task 1 again: Solve using scipy.linalg.solve
    x_scipy, time_scipy = solve_system(A, b, use_scipy=True)
    print(f"Solution using scipy.linalg.solve: {x_scipy}, Time taken: {time_scipy:.6f} seconds")
    # Comment: Both numpy.linalg.solve and scipy.linalg.solve are similar in their functionality, 
    # and both use LU decomposition for solving dense matrices. However, scipy.linalg.solve offers 
    # more flexibility for advanced matrix types such as sparse matrices, while numpy.linalg.solve is 
    # primarily designed for dense matrices.
    
    # Comparing the time for this task, we can see that numpy.linalg.solve is about 10x faster
    # consistently. 
    
    # Task 2: Solve using inverse of A
    x_inv, time_inv = solve_system(A, b, method='inv')
    print(f"Solution using inverse method: {x_inv}, Time taken: {time_inv:.6f} seconds")
    
    # Computing the inverse explicitly is less efficient, more computationally expensive, and 
    # can be numerically unstable, especially for large or ill-conditioned matrices. In practice, 
    # solving systems by computing the inverse is discouraged because direct methods (such as LU 
    # decomposition - the next one) are faster and more accurate. This is what I've read online,
    # but we see slightly faster performance on inv comapred to LU for this case (possibly because
    # this matrix-vector pair is small?)
    
    # Task 3: Solve using LU decomposition
    x_lu, time_lu = solve_system(A, b, method='lu')
    print(f"Solution using LU decomposition: {x_lu}, Time taken: {time_lu:.6f} seconds")
    
    # LU decomposition is superior when solving multiple systems with the same matrix A 
    # but different right-hand side vectors b. Once the matrix is decomposed, solving for 
    # each b becomes more efficient compared to recomputing the solution from scratch each time.
     
    # Task 4: Calculate the condition number of A
    cond_number = np.linalg.cond(A)
    print(f"Condition number of A: {cond_number:.6f}")
    
    # The condition number gives an estimate of how sensitive the solution is to changes 
    # in the matrix A. A high condition number indicates that the system is ill-conditioned, 
    # meaning small changes in input or rounding errors may lead to large errors in the solution. 
    # A low condition number indicates a well-conditioned system, meaning it is more stable and 
    # less sensitive to such changes.
    
    # Additional test cases
    print("\n--- Additional Tests ---")

    # Test Case 1
    A1 = np.array([[3, 18, 9], [2, 3, 3], [4, 1, 2]])
    b1 = np.array([18, 117, 283])
    x1, time1 = solve_system(A1, b1)
    print(f"Test Case 1 Solution: {x1}, Time taken: {time1:.6f} seconds")
    # I know this is correct because I checked by hand. Also it's well conditioned and solves
    # quickly.
    
    # Test Case 2
    A2 = np.array([[20, 15, 10], [-3, -2.24999, 7], [5, 1, 3]])
    b2 = np.array([45, 1.751, 9])
    x2, time2 = solve_system(A2, b2)
    print(f"Test Case 2 Solution: {x2}, Time taken: {time2:.6f} seconds")
    # The matrix A2 is slightly ill-conditioned (condition number ≈12.87), which might lead to small
    # numerical inaccuracies. However, the result is still pretty close to the expected solution [1,1,1]. 
    # If the condition number were significantly higher, we would expect much larger deviations in the
    # result due to numerical instability.
