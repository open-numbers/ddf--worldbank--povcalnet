import polars as pl
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from functools import partial


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
        window = int(window_func(i))
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
            filtered_value = savgol_filter(
                segment, len(segment), polyorder, mode="nearest"
            )
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

data.filter(
    (pl.col('i') <= 200) & (pl.col('headcount') != 0)
)


def _f(df, **kwargs):
    return df.filter(pl.all_horizontal([(pl.col(k) == v) for k, v in kwargs.items()]))


df = _f(data, 
        country="TUR", 
        year=2008, 
        reporting_level="national")
df["headcount"].diff().arg_max()
df[299]


def window_func(pos, max_window, dense_left, dense_right, total_x=461):
    """
    """
    min_window = int((dense_right - dense_left) / 5)
    if min_window < 3:
        min_window = 3

    if pos < dense_left:  # Exponential decay region
        # Calculate decay constant
        # We want: max_window * exp(-k * 0.3) = min_window
        # Therefore: k = -ln(min_window/max_window) / 0.3
        k = -np.log(min_window / max_window) / dense_left
        window = max_window * np.exp(-k * pos)

    elif pos > dense_right:  # Exponential growth region
        # Similar calculation but for growth from 0.7 to 1.0
        right_remaining = total_x - dense_right
        k = np.log(max_window / min_window) / right_remaining
        window = min_window * np.exp(k * (pos - dense_right))

    else:  # Constant region
        window = min_window

    # Ensure window size is odd
    window = int(round(window))
    if window > max_window:
        window = max_window
    if window % 2 == 0:
        window += 1

    return window


poss = np.arange(0, 500, 1)
ws = np.array([window_func(x, 100, 238, 328) for x in poss])

plt.plot(poss, ws)
plt.show()


# function to detect the shape parameters for noisy shapes
def fwhw(y):
    dif = y.max() - y.min()
    hm = dif / 2
    nearest = (np.abs(y - hm)).argmin()
    middle = y.argmax()
    width = np.abs(middle - nearest) * 2
    print(nearest, middle, width)
    if middle > nearest:
        return nearest, nearest + width
    else:
        return nearest - width, nearest


xlocs = fwhw(np.diff(df['headcount']))
xlocs

xlocs[1] - xlocs[0]

plt.plot(df['headcount'].diff())
plt.vlines(xlocs, 0, df['headcount'].diff().max(), linestyles='dashed')
plt.show()



left, right = fwhw(np.diff(df['headcount']))
left, right

wf = partial(window_func, max_window=150, dense_left=left, dense_right=right)

y_ = variable_savgol_filter(df["headcount"], wf, polyorder=3)
y_ = np.maximum.accumulate(y_)
y_ = np.clip(y_, 0, 1)

for i in range(20):
    y_ = variable_savgol_filter(y_, wf, polyorder=3)
    y_ = np.maximum.accumulate(y_)
    y_ = np.clip(y_, 0, 1)


plt.plot(df["i"], df["headcount"])
plt.plot(df["i"], y_)
plt.show()

d = df["headcount"] - y_
plt.plot(d)
plt.vlines(200, d.min(), d.max(), color="blue", linestyles="dashed", alpha=0.5)
plt.show()

plt.plot(np.diff(df["headcount"]))
plt.plot(np.diff(y_))
plt.show()

d[200]


# NEXT: let's check more examples?
# I think it's already good enough!
