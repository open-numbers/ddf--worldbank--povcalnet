import polars as pl
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt


def variable_savgol_filter(y, window_func, polyorder=2):
    """
    Apply Savitzky-Golay filter with variable window size.

    Parameters:
    -----------
    y : array-like
        Input signal to be filtered
    window_func : callable
        Function that takes index position (0 to 1) and returns window size
        Should return odd integers
    polyorder : int
        Order of polynomial to fit

    Returns:
    --------
    numpy.ndarray
        Filtered signal
    """
    y = np.array(y)
    n_points = len(y)
    y_filtered = np.zeros(n_points)

    # Apply filter for each point with custom window
    for i in range(n_points):
        # Get window size for this position
        window = int(window_func(y[i]))
        if window % 2 == 0:
            window += 1  # Ensure odd window size

        # Handle edge cases where window would extend beyond signal
        half_window = window // 2
        if i < half_window:
            window = 2 * i + 1
        elif i >= n_points - half_window:
            window = 2 * (n_points - i - 1) + 1

        # Apply Savitzky-Golay filter with computed window
        start_idx = max(0, i - window // 2)
        end_idx = min(n_points, i + window // 2 + 1)
        segment = y[start_idx:end_idx]

        if len(segment) >= polyorder + 1:
            filtered_value = savgol_filter(segment, len(segment), polyorder, mode='nearest')
            y_filtered[i] = filtered_value[len(segment) // 2]
        else:
            y_filtered[i] = y[i]  # Use original value if window too small

    return y_filtered


# Example usage:
def demo_variable_savgol():
    # Generate sample noisy data
    x = np.linspace(0, 10, 1000)
    y_true = np.sin(x) + 0.5 * np.sin(2 * x)
    noise = np.random.normal(0, 0.1, len(x))
    y_noisy = y_true + noise

    # Define window size function - larger at edges, smaller in middle
    def window_func(pos):
        # Parabolic window size variation: larger at edges, smaller in middle
        base_window = 51  # minimum window size
        extra_window = 100  # additional window size at edges
        return int(base_window + extra_window * (2 * pos - 1) ** 2)

    # Apply variable window filter
    y_filtered = variable_savgol_filter(y_noisy, window_func)

    return x, y_true, y_noisy, y_filtered


x, y_true, y_noisy, y_filtered = demo_variable_savgol()


plt.figure(figsize=(12, 6))
plt.plot(x, y_noisy, "gray", alpha=0.5, label="Noisy signal")
plt.plot(x, y_true, "k--", label="True signal")
plt.plot(x, y_filtered, "r", label="Filtered signal")
plt.legend()
plt.title("Variable Window Savitzky-Golay Filter")
plt.show()


# let's check create a window function for CDF
data = pl.read_parquet("../build/povcalnet_clean.parquet")
data


def _f(df, **kwargs):
    return df.filter(pl.all_horizontal([(pl.col(k) == v) for k, v in kwargs.items()]))


df = _f(data, country='AGO', year=2000, reporting_level='national')
df['headcount'].diff().arg_max()
df[226]


def window_func(pos):
    """
    Create exponential window size variation:
    - x in [0, 0.3]: exponential decay from 100 to 3
    - x in [0.3, 0.7]: constant at 3
    - x in [0.7, 1.0]: exponential growth from 3 to 100
    
    Parameters:
    -----------
    pos : float
        Position in signal (0 to 1)
        
    Returns:
    --------
    int
        Window size (odd integer)
    """
    # Constants
    max_window = 100
    min_window = 11
    
    if pos < 0.3:  # Exponential decay region
        # Calculate decay constant
        # We want: max_window * exp(-k * 0.3) = min_window
        # Therefore: k = -ln(min_window/max_window) / 0.3
        k = -np.log(min_window/max_window) / 0.3
        window = max_window * np.exp(-k * pos)
        
    elif pos > 0.7:  # Exponential growth region
        # Similar calculation but for growth from 0.7 to 1.0
        k = np.log(max_window/min_window) / 0.3
        window = min_window * np.exp(k * (pos - 0.7))
        
    else:  # Constant region
        window = min_window
    
    # Ensure window size is odd
    window = int(round(window))
    if window % 2 == 0:
        window += 1
        
    return window


poss = np.linspace(0, 1, 1000)
ws = np.array([window_func(x) for x in poss])

plt.plot(poss, ws)
plt.show()


y_ = variable_savgol_filter(df['headcount'], window_func, polyorder=3)
y_ = np.maximum.accumulate(y_)
y_ = np.clip(y_, 0, 1)

for i in range(3):
    y_ = variable_savgol_filter(y_, window_func, polyorder=3)
    y_ = np.maximum.accumulate(y_)
    y_ = np.clip(y_, 0, 1)


plt.plot(df['i'], df['headcount'])
plt.plot(df['i'], y_)
plt.show()

d = df['headcount'] - y_
plt.plot(d)
plt.vlines(200, d.min(), d.max(), color='blue', linestyles='dashed', alpha=.5)
plt.show()

plt.plot(np.diff(df['headcount']))
plt.plot(np.diff(y_))
plt.show()

d[200]


# NEXT: try to use adaptive method to findout the correct window size
def mle_normal_parameters(observations):
    pass


mle_normal_parameters(np.diff(y_))

df.select(
    pl.col('i'),
    pl.col('headcount').diff()
).write_csv('~/Downloads/test.csv')


