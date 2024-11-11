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
        country="PAN",
        year=1992,
        reporting_level="national")
df["headcount"].diff().arg_max()
df[299]

plt.plot(df['headcount'])
plt.show()


def window_func(pos, max_window, dense_left, dense_right, total_x=461, min_window=3):
    """
    """
    # the min window based on data
    min_window_ = int((dense_right - dense_left) / 5)
    if min_window_ > min_window:
        min_window = min_window_

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
ws = np.array([window_func(x, 200, 206, 214) for x in poss])

plt.plot(poss, ws)
plt.show()


# we assume that the top of the shapes must be near the middle of the shape
# not in the tails. So we just reset the tails to 0 to calculate fwhw
def remove_tails(y, a, b):
    y[:a] = 0
    y[b:] = 0
    return y


def find_fwhm_range(cdf_values, lower=0.3, upper=0.7):
    # Find indices where CDF is between lower and upper bounds
    mask = (cdf_values >= lower) & (cdf_values <= upper)
    indices = np.where(mask)[0]

    # Get the corresponding CDF values
    values_in_range = cdf_values[mask]

    return indices, values_in_range


idxs, _ = find_fwhm_range(df['headcount'].to_numpy())
a = idxs[0]
b = idxs[-1]


# function to detect the shape parameters for noisy shapes
def fwhm(y, a, b):
    y = remove_tails(y, a, b)
    # Calculate half maximum
    peak_height = y.max() - y.min()
    half_max = y.min() + peak_height / 2

    # Find peak position
    middle = y.argmax()

    # Function to check if 10 consecutive points are below half max
    def check_consecutive_points(start_idx, direction=1):
        consecutive_count = 0
        idx = start_idx

        while 0 <= idx < len(y):
            if y[idx] < half_max:
                consecutive_count += 1
                if consecutive_count >= 10:
                    # Return the first point where it went below half max
                    return idx - (4 if direction > 0 else 0)
            else:
                consecutive_count = 0
            idx += direction

        return None

    # Search for left boundary (moving backwards from peak)
    left_boundary = check_consecutive_points(middle, direction=-1)

    # Search for right boundary (moving forwards from peak)
    right_boundary = check_consecutive_points(middle, direction=1)

    # Handle edge cases where boundaries aren't found
    if left_boundary is None:
        left_boundary = 0
    if right_boundary is None:
        right_boundary = len(y) - 1
    # width = right_boundary - left_boundary
    return left_boundary, right_boundary


input_shape = np.diff(df['headcount'])
xlocs = fwhm(input_shape, a, b)
xlocs

xlocs[1] - xlocs[0]

plt.plot(df['headcount'].diff())
plt.vlines(xlocs, 0, df['headcount'].diff().max(), linestyles='dashed')
plt.show()


left, right = fwhm(np.diff(df['headcount']), a, b)
left, right

wf = partial(window_func, max_window=80, dense_left=left, dense_right=right)

y_ = variable_savgol_filter(df["headcount"], wf, polyorder=1)
y_ = np.maximum.accumulate(y_)
y_ = np.clip(y_, 0, 1)

for i in range(20):
    y_ = variable_savgol_filter(y_, wf, polyorder=1)
    y_ = np.maximum.accumulate(y_)
    y_ = np.clip(y_, 0, 1)


plt.plot(df["i"], df["headcount"])
plt.plot(df["i"], y_)
plt.show()


# fix the beginning. we assume that in the beginning the shape will be monotonly increasing
def fix_head(y, a):
    y[:a] = np.sort(y[:a])
    return y


d = df["headcount"] - y_
plt.plot(d)
plt.vlines(200, d.min(), d.max(), color="blue", linestyles="dashed", alpha=0.5)
plt.show()

plt.plot(np.diff(df["headcount"]))
plt.plot(fix_head(np.diff(y_), a))
# plt.plot(np.diff(y_))
plt.show()

d[200]

r = fix_head(np.diff(y_), a)
r.sum()


def create_smooth_pdf_shape_(noisy_cdf):
    idxs, _ = find_fwhm_range(noisy_cdf)
    a = idxs[0]
    b = idxs[-1]

    left, right = fwhm(np.diff(noisy_cdf), a, b)

    # first use polyorder = 1 to reduce noise
    wf = partial(window_func, max_window=40, dense_left=left, dense_right=right)

    y = noisy_cdf
    for i in range(2):
        y = variable_savgol_filter(y, wf, polyorder=1)
        y = np.clip(y, 0, 1)

    # then, use polyorder = 2 to smooth the shape
    wf = partial(window_func, max_window=80, dense_left=left, dense_right=right,
                 min_window=7)

    for i in range(10):
        y = variable_savgol_filter(y, wf, polyorder=2)
        y = np.clip(y, 0, 1)

    # ensure the monotone
    y = np.maximum.accumulate(y)

    return y


# more examples
df = _f(data,
        country="CHN",
        year=1983,
        reporting_level="national")
df["headcount"].diff().arg_max()

y_ = create_smooth_pdf_shape_(df['headcount'].to_numpy())

ttt = np.diff(y_)
np.max(ttt[:a]) > np.max(ttt[a:])

plt.plot(np.diff(df["headcount"]))
# plt.plot(fix_head(np.diff(y_), a))
plt.plot(np.diff(y_))
plt.show()

plt.plot(df["i"], df["headcount"])
plt.plot(df["i"], y_)
plt.show()

# the above procedure should be optimal for smoothing in most cases.
# FIXME: SWZ - 1983 is a special case, see if we need to handle it or report to WB.
