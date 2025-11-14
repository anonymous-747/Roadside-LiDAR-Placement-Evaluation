import numpy as np
from scipy.integrate import dblquad
from numba import jit

# --- Global Constants ---
# These are the bounds for the variables used in the distributions.
upper_threshold = 2
lower_threshold = 0.5
upper_threshold2 = 2
lower_threshold2 = 0.5  


@jit(nopython=True)
def _numba_integrand_fixed(y, x, n, m, L, L2, IOU):
    """
    Numba-accelerated integrand for the first double integral.

    This function encodes:
      1. A joint PDF for (x, y) whose form depends on parameters n and m.
      2. A condition involving the IOU constraint. The integrand returns
         the joint PDF if the condition is satisfied, otherwise 0.

    Args:
        y (float): Integration variable y (inner integral).
        x (float): Integration variable x (outer integral).
        n (int or float): Shape parameter for the x distribution.
        m (int or float): Shape parameter for the y distribution.
        L (float): Lower bound for x (≥ lower_threshold).
        L2 (float): Lower bound for y (≥ lower_threshold).
        IOU (float): IOU threshold used in the condition.

    Returns:
        float: The value of the integrand at (x, y).
    """
    # 1. Calculate the joint PDF based on the values of n and m
    joint_pdf_val = 0.0

    # Case 1: x ~ 1/x (log-uniform), y ~ 1/y (log-uniform)
    if n == 1 and m == 1:
        norm_const = np.log(upper_threshold / L) * np.log(upper_threshold / L2)
        # Only valid if the normalization constant is positive
        if norm_const > 0:
            joint_pdf_val = 1 / (x * y * norm_const)

    # Case 2: x ~ 1/x (log-uniform), y ~ power-law (m > 1)
    elif n == 1 and m > 1:
        log_term_x = np.log(upper_threshold / L)
        c_y_denom = (np.power(L2, 1 - m) - np.power(upper_threshold, 1 - m))
        if log_term_x > 0 and c_y_denom != 0:
            C_y = (m - 1) / c_y_denom
            joint_pdf_val = (1 / (x * log_term_x)) * (C_y / np.power(y, m))

    # Case 3: x ~ power-law (n > 1), y ~ 1/y (log-uniform)
    elif n > 1 and m == 1:
        c_x_denom = (np.power(L, 1 - n) - np.power(upper_threshold, 1 - n))
        log_term_y = np.log(upper_threshold / L2)
        if c_x_denom != 0 and log_term_y > 0:
            C_x = (n - 1) / c_x_denom
            joint_pdf_val = (C_x / np.power(x, n)) * (1 / (y * log_term_y))

    # Case 4: x ~ power-law (n > 1), y ~ power-law (m > 1)
    else:
        c_x_denom = (np.power(L, 1 - n) - np.power(upper_threshold, 1 - n))
        c_y_denom = (np.power(L2, 1 - m) - np.power(upper_threshold, 1 - m))
        if c_x_denom != 0 and c_y_denom != 0:
            C_x = (n - 1) / c_x_denom
            C_y = (m - 1) / c_y_denom
            joint_pdf_val = (C_x * C_y) / (np.power(x, n) * np.power(y, m))

    # 2. Apply the IOU-based condition
    #    condition_val corresponds to some transformed overlap quantity.
    condition_val = (1 + IOU) * (min(x, 1.0) * min(y, 1.0)) - IOU * x * y

    # If the condition is satisfied, contribute the PDF; otherwise, 0
    if condition_val > IOU:
        return joint_pdf_val
    else:
        return 0.0


def calculate_integral_fixed(n, m, a_max, b_max, IOU):
    """
    Compute the integral of the joint PDF under an IOU constraint.

    This function:
      1. Derives effective lower bounds L and L2 from (a_max, b_max).
      2. Checks if the integration region is valid.
      3. Uses SciPy's dblquad with a Numba-accelerated integrand.

    Args:
        n (int or float): Shape parameter for x.
        m (int or float): Shape parameter for y.
        a_max (float): Lower bound candidate for x; clamped to at least lower_threshold.
        b_max (float): Lower bound candidate for y; clamped to at least lower_threshold.
        IOU (float): IOU threshold used inside the integrand condition.

    Returns:
        (result, error):
            result (float): Value of the double integral.
            error (float): Estimated numerical integration error.
    """
    # Clamp a_max, b_max to at least lower_threshold
    L = np.maximum(lower_threshold, a_max)
    L2 = np.maximum(lower_threshold, b_max)

    # If the lower bound reaches or exceeds the upper bound, the region is empty
    if L >= upper_threshold or L2 >= upper_threshold:
        return 0.0, 0.0

    # Perform the double integration over x in [L, upper_threshold]
    # and y in [L2, upper_threshold]
    result, error = dblquad(
        _numba_integrand_fixed,
        L,                      # x lower bound
        upper_threshold,        # x upper bound
        lambda x: L2,           # y lower bound
        lambda x: upper_threshold,
        args=(n, m, L, L2, IOU) # Extra arguments passed to _numba_integrand_fixed
    )
    
    return result, error


@jit(nopython=True)
def _numba_integrand(w, x, n_param, r_param, mid_param, IOU):
    """
    Numba-accelerated integrand for the second double integral.

    This function:
      1. Checks an IOU-based condition for a 1D overlap configuration
         (using center 'mid_param' and width 'w' against x).
      2. If the condition is satisfied, evaluates p_x(x) * p_w(w).
      3. Otherwise returns 0.

    Args:
        w (float): Integration variable for the width parameter.
        x (float): Integration variable for the position parameter.
        n_param (float): Shape parameter controlling the distribution of w.
        r_param (float): Minimum allowed value of w; enforced to be ≥ lower_threshold2.
        mid_param (float): Center position used in overlap calculation.
        IOU (float): IOU threshold used in the condition.

    Returns:
        float: Integrand value at (x, w).
    """
    # Enforce a minimum bound for r_param
    r_param = max(lower_threshold2, r_param)

    # 1. Condition check based on the overlap geometry
    # Compute the overlap length in [0, 1] domain
    term1 = min(mid_param + w / 2.0, 1.0)
    term2 = max(mid_param - w / 2.0, 0.0)
    term3 = min(x, 1.0)
    # 'val' models the overlap or intersection component
    val = (term1 - term2) * term3
    
    # Total "union-like" denominator for the IOU expression
    denominator = x * w + 1.0 - val

    # Avoid division by zero or extremely small denominators
    if abs(denominator) < 1e-9: 
        return 0.0
    
    # Condition derived from IOU > IOU_threshold
    is_condition_met = (val - IOU * denominator > 0)
    
    if not is_condition_met:
        return 0.0

    # 2. PDF calculations (only if condition is satisfied)

    # p_x(x): Uniform distribution over [lower_threshold2, upper_threshold2]
    if lower_threshold2 <= x <= upper_threshold2:
        p_x_val = 1.0 / (upper_threshold2 - lower_threshold2)
    else:
        p_x_val = 0.0
    
    # p_w(w): Either uniform or power-law depending on n_param
    p_w_val = 0.0

    # Case 1: Uniform distribution in w
    if n_param == 1:
        p_w_val = 1.0 / (upper_threshold2 - lower_threshold2)
    else:
        # Case 2: Power-law distribution in w on [r_param, upper_threshold2]
        if r_param <= w <= upper_threshold2:
            # Normalization denominator for the power-law
            norm_denom = r_param**(1.0 - n_param) - upper_threshold2**(1.0 - n_param)
            if abs(norm_denom) > 1e-9:
                normalization_factor = (n_param - 1.0) / norm_denom
                p_w_val = normalization_factor / (w**n_param)

    return p_x_val * p_w_val


def calculate_integral(R, n, mid, IOU):
    """
    Compute the second double integral using the Numba-accelerated integrand.

    Integration domain:
      x ∈ [lower_threshold2, upper_threshold2]
      w ∈ [R_clamped, upper_threshold2]

    where R_clamped = max(lower_threshold2, R).

    Args:
        R (float): Lower bound for w (will be clamped to ≥ lower_threshold2).
        n (float): Shape parameter for the w distribution.
        mid (float): Center value used in overlap/IOU computation.
        IOU (float): IOU threshold passed into the integrand.

    Returns:
        (result, error):
            result (float): Value of the double integral.
            error (float): Estimated numerical integration error.
    """
    # Ensure R is not less than the global lower bound
    R = max(lower_threshold2, R)

    # x integration bounds
    x_min, x_max = lower_threshold2, upper_threshold2
    # w integration bounds
    w_min, w_max = R, upper_threshold2

    # Note: dblquad integrates over x first, and w second when passing
    #       (func, x_min, x_max, w_min, w_max, ...),
    #       so the integrand signature is (w, x, ...).
    result, error = dblquad(
        _numba_integrand, 
        x_min, 
        x_max, 
        w_min, 
        w_max, 
        args=(n, R, mid, IOU)
    )

    return result, error


# --- Example Usage ---
# You can call these functions like:
#   val1, err1 = calculate_integral_fixed(n=1, m=2, a_max=0.7, b_max=0.8, IOU=0.5)
#   val2, err2 = calculate_integral(R=0.6, n=2.0, mid=0.5, IOU=0.7)
# Make sure SciPy and Numba are installed and properly configured.
