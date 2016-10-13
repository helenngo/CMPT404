import numpy as np
import copy
 
def  gen_error(N0=1000):
    while True:
        # Equation (2.13)
        N = (8/0.05**2)*np.log((4*(2*N0)**10+4)/0.05)
        # Make N an integer > N
        N = np.ceil(N)
        # Test for convergence
        if N/N0 == 1:
            break
        # Update with new guess
        N0 = copy.deepcopy(N)
    # Return sample size
    return N

        
def main():
    # Guess samle size of 1000
    sample_size = gen_error(N0=1000)
    print(sample_size)
main()
