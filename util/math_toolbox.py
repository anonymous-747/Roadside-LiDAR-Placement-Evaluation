import numpy as np
from scipy.integrate import dblquad
from numba import jit

# --- Global Constants ---
upper_threshold = 2
lower_threshold = 0.5
upper_threshold2 = 2
lower_threshold2 = 0.5  


@jit(nopython=True)
def _numba_integrand_fixed(y, x, n, m, L, L2, IOU):
    """
    Numba-accelerated JIT helper function.
    This contains the core logic that is repeatedly called by the integrator.
    All conditional logic is moved inside this single function for efficient compilation.
    """
    # 1. Calculate the joint PDF based on the values of n and m
    joint_pdf_val = 0.0
    if n == 1 and m == 1:
        norm_const = np.log(upper_threshold / L) * np.log(upper_threshold / L2)
        if norm_const > 0: # Avoid division by zero
            joint_pdf_val = 1 / (x * y * norm_const)
            
    elif n == 1 and m > 1:
        log_term_x = np.log(upper_threshold / L)
        # Denominator for C_y pre-calculation
        c_y_denom = (np.power(L2, 1 - m) - np.power(upper_threshold, 1 - m))
        if log_term_x > 0 and c_y_denom != 0:
            C_y = (m - 1) / c_y_denom
            joint_pdf_val = (1 / (x * log_term_x)) * (C_y / np.power(y, m))
            
    elif n > 1 and m == 1:
        # Denominator for C_x pre-calculation
        c_x_denom = (np.power(L, 1 - n) - np.power(upper_threshold, 1 - n))
        log_term_y = np.log(upper_threshold / L2)
        if c_x_denom != 0 and log_term_y > 0:
            C_x = (n - 1) / c_x_denom
            joint_pdf_val = (C_x / np.power(x, n)) * (1 / (y * log_term_y))
            
    else:  # n > 1 and m > 1
        c_x_denom = (np.power(L, 1 - n) - np.power(upper_threshold, 1 - n))
        c_y_denom = (np.power(L2, 1 - m) - np.power(upper_threshold, 1 - m))
        if c_x_denom != 0 and c_y_denom != 0:
            C_x = (n - 1) / c_x_denom
            C_y = (m - 1) / c_y_denom
            joint_pdf_val = (C_x * C_y) / (np.power(x, n) * np.power(y, m))

    # 2. Apply the condition
    condition_val = (1 + IOU) * (min(x, 1.0) * min(y, 1.0)) - IOU * x * y
    
    if condition_val > IOU:
        return joint_pdf_val
    else:
        return 0.0

def calculate_integral_fixed(n, m, a_max, b_max, IOU):
    """
    Calculates the integral of the joint PDF, accelerated with Numba.
    """
    L = np.maximum(lower_threshold, a_max)
    L2 = np.maximum(lower_threshold, b_max)

    if L >= upper_threshold or L2 >= upper_threshold:
        return 0.0, 0.0

    # Perform the double integration using the JIT-compiled helper function
    result, error = dblquad(
        _numba_integrand_fixed,
        L,
        upper_threshold,
        lambda x: L2,
        lambda x: upper_threshold,
        args=(n, m, L, L2, IOU) # Pass additional parameters to the Numba function
    )
    
    return result, error


@jit(nopython=True)
# CHANGED: Removed the default IOU=0.7 from the signature
def _numba_integrand(w, x, n_param, r_param, mid_param, IOU):
    """
    Numba-accelerated JIT helper function for the second integral.
    This contains all the logic repeatedly called by the integrator.
    """
    r_param = max(lower_threshold2,r_param)

    # 1. Condition check (inlined for performance)
    # Value calculation
    term1 = min(mid_param + w / 2.0, 1.0)
    term2 = max(mid_param - w / 2.0, 0.0)
    term3 = min(x, 1.0)
    val = (term1 - term2) * term3
    
    # Denominator check
    denominator = x * w + 1.0 - val
    # A small epsilon is safer than a direct zero check in floating point math
    if abs(denominator) < 1e-9: 
        return 0.0
    
    # The condition now uses the IOU passed as an argument
    is_condition_met = (val -IOU* denominator > 0)
    
    if not is_condition_met:
        return 0.0

    # 2. PDF Calculations (if condition is met)
    # p_x(x)
    p_x_val = 1.0/(upper_threshold2-lower_threshold2) if lower_threshold2 <= x <= upper_threshold2 else 0.0
    
    # p_w(w)
    p_w_val = 0.0

    if (n_param == 1):
        p_w_val = 1.0/(upper_threshold2-lower_threshold2)
    else:
        if r_param <= w <= upper_threshold2:
            # Denominator of normalization factor
            norm_denom = r_param**(1.0 - n_param) - upper_threshold2**(1.0 - n_param)
            if abs(norm_denom) > 1e-9: # Avoid division by zero
                normalization_factor = (n_param - 1.0) / norm_denom
                p_w_val = normalization_factor / (w**n_param)

    return p_x_val * p_w_val

# CHANGED: Added IOU to the function signature
def calculate_integral(R, n, mid, IOU):
    """
    Calculates the specified double integral, accelerated with Numba.
    """
    R=max(lower_threshold2,R)
    x_min, x_max = lower_threshold2, upper_threshold2
    w_min, w_max = R, upper_threshold2

    result, error = dblquad(
        _numba_integrand, 
        x_min, 
        x_max, 
        w_min, 
        w_max, 
        # CHANGED: Pass the new IOU argument to the integrand
        args=(n, R, mid, IOU)
    )

    return result, error


# --- Example Usage ---