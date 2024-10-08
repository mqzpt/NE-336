# Assignment 1 - Question 2
# Matthew Athanasopoulos

def summation(x, tolerance=1e-5):
    """
    Evaluates the infinite series sum for f(x) = sum(x^(2n+1)) from n = 0 to infinity,
    which is valid for |x| < 1.

    This function calculates the summation iteratively until the absolute approximate 
    relative error is smaller than the specified tolerance.

    Parameters:
    x (float): The input value for which the series is to be evaluated. Must satisfy |x| < 1.
    tolerance (float, optional): The tolerance for the approximate relative error. Defaults to 1e-5.

    Returns:
    float: The calculated sum of the series if input is valid.
    None: If the input is invalid (either not a number or |x| >= 1), returns None and prints an error message.
    """

    # Check if input is a number
    if not isinstance(x, (float, int)):
        print("Error: Input must be a number.")
        return None

    # Check if |x| < 1
    if abs(x) >= 1:
        print("Error: |x| must be less than 1.")
        return None

    # Initialize variables
    n = 0
    term = x  # First term of the series is x^1
    sum_value = term
    error = abs(term)

    # Summation loop with approximate relative error check
    while error > tolerance:
        n += 1
        term = x**(2*n + 1)  # Update term to x^(2n+1)
        sum_value += term
        # Calculate absolute error (error in current term)
        error = abs(term/sum_value)

    return sum_value

# Example test cases


# Testing Invalid Inputs
print('Testing Invalid Inputs:')
print('\nThe first example is an input of 1.2, which has an absolute value greater than 1. Output of summation(1.2) is below:')
summation(1.2)
print('\nThe first example is a string input, which is not a float or int. Output of summation(\'hello world\') is below:')
summation('hello world')

# Testing Valid Inputs
print("\nTesting valid case:")
x = 0.7
result = summation(x)
print(f"\nSummation for x = {x} is approximately: {result}")
