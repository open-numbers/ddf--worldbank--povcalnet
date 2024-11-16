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


data = pl.read_parquet("../build/povcalnet_clean.parquet")
data

data.filter(
    pl.col('i') <= 200, pl.col('headcount') != 0
)


def _f(df, **kwargs):
    return df.filter(pl.all_horizontal([(pl.col(k) == v) for k, v in kwargs.items()]))


df = _f(data,
        country="PAN",
        year=1992,
        reporting_level="national")
df


# let's start with removing extreme points
# and then ensure the shape sums to 1 before smoothing.

pdf = df.select(
    pl.col('i').shift(1),
    pl.col('headcount').diff(),
).drop_nulls()
pdf = pdf.to_pandas()
pdf['mean'] = pdf['headcount'].rolling(3, min_periods=1, center=True).mean()
pdf.iloc[100:150]

m = pdf['headcount'] - pdf['mean']
mask = m > 4 * m.std()
m.plot()
plt.show()


plt.plot(pdf[~mask]['headcount'])
plt.scatter(pdf[mask]['i'], pdf[mask]['headcount'])
plt.show()


def display_outliners(country, year, reporting_level='national', p=4):
    df = _f(data,
            country=country,
            year=year,
            reporting_level=reporting_level)
    pdf = df.select(
        pl.col('i').shift(1),
        pl.col('headcount').diff(),
    ).drop_nulls()
    pdf = pdf.to_pandas()
    pdf['mean'] = pdf['headcount'].rolling(3, min_periods=1, center=True).mean()
    m = pdf['headcount'] - pdf['mean']
    mask = m > p * m.std()
    plt.plot(pdf[~mask]['headcount'])
    plt.scatter(pdf[mask]['i'], pdf[mask]['headcount'])
    plt.show()


display_outliners("USA", 2012)


def create_pdf_and_remove_outliners(df: pl.DataFrame, p: int):
    pdf = df.select(
        pl.col('i').shift(1),
        pl.col('headcount').diff(),
    ).drop_nulls()
    pdf = pdf.to_pandas()
    pdf['mean'] = pdf['headcount'].rolling(3, min_periods=1, center=True).mean()
    m = pdf['headcount'] - pdf['mean']
    mask = m > p * m.std()

    # set the masked to null
    pdf.loc[mask, 'headcount'] = None
    pdf['headcount'] = pdf['headcount'].interpolate('linear', limit_direction='both')
    return pl.DataFrame(pdf[['i', 'headcount']])


df
pdf = create_pdf_and_remove_outliners(df, 4)
pdf

# because we want to also keep the peak position,
# we don't want to spread all points equally into the remaining shape.
pdf_ = pdf.select(
    pl.col('headcount') / pl.col('headcount').sum()
)

plt.plot(pdf['headcount'])
plt.plot(pdf_['headcount'])
plt.show()


# use a better way, first we get some weights
def generate_weights(values):
    """
    Generate weights from an array of non-negative values where:
    - Zero values get zero weights
    - Non-zero values get weights proportional to their difference from the maximum value
    - Values to the left of the maximum get 4x of weights because the noise usually happend on the left.
    - All weights sum to 1

    Parameters:
    values (array-like): 1D array of non-negative numbers

    Returns:
    numpy.ndarray: Array of weights that sum to 1
    """
    # Convert input to numpy array and verify non-negative values
    values = np.array(values)
    if np.any(values < 0):
        raise ValueError("All values must be non-negative")

    # Create mask for non-zero values
    non_zero_mask = values > 0

    # Initialize weights array with zeros
    weights = np.zeros_like(values, dtype=float)

    if np.any(non_zero_mask):
        # Get maximum value and its position
        max_position = np.argmax(values)

        # Create position bias multiplier (2 for left side, 1 for right side)
        position_multiplier = np.ones_like(values)
        position_multiplier[:max_position] = 1.2  # more weights for left side

        # Apply position multiplier to differences
        weighted_differences = values * position_multiplier

        # For non-zero values, use the weighted differences
        valid_mask = non_zero_mask
        valid_differences = weighted_differences[valid_mask]

        # If all differences are 0 (all values are equal), use equal weights
        if np.all(valid_differences == 0):
            weights[valid_mask] = 1 / np.sum(valid_mask)
        else:
            # Normalize the differences to get weights
            weights[valid_mask] = valid_differences / np.sum(valid_differences)

    return weights


generate_weights(pdf['headcount'].to_numpy())


weights = pdf.select(
    pl.col('headcount').map_batches(generate_weights)
)

plt.plot(weights['headcount'])
plt.show()


pdf_ = pdf.select(
    (1 - pl.col('headcount').sum()) * pl.col('headcount').map_batches(generate_weights) + pl.col('headcount')
)

pdf_['literal'].sum()

plt.plot(pdf_['literal'])
plt.plot(pdf['headcount'])
plt.show()

# also check the cdf
plt.plot(pdf_['literal'].cum_sum())
plt.plot(df['headcount'][1:])
plt.show()

# test more examples
df = _f(data,
        country="PAN",
        year=1996,
        reporting_level="national")

pdf = create_pdf_and_remove_outliners(df, 3.29)
pdf_ = pdf.select(
    (1 - pl.col('headcount').sum()) * pl.col('headcount').map_batches(generate_weights) + pl.col('headcount')
)

# looks good
# next we convert it back to CDF and continue to do smoothing.
# use CDF for smoothing is easier because its monotone
cdf_clean = pdf_.select(
    pl.lit(0).append(pl.col('literal')).cum_sum().alias('headcount')
).with_row_index('i')


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


# fix the beginning. we assume that in the beginning the shape will be monotonly increasing
def fix_head(y, a):
    y[:a] = np.sort(y[:a])
    return y


def create_smooth_pdf_shape_(noisy_cdf):
    idxs, _ = find_fwhm_range(noisy_cdf)
    a = idxs[0]
    b = idxs[-1]

    left, right = fwhm(np.diff(noisy_cdf), a, b)

    # first try to reduce noise
    wf = partial(window_func, max_window=40, dense_left=left, dense_right=right, 
                 min_window=1)

    y = noisy_cdf
    for i in range(2):
        y = variable_savgol_filter(y, wf, polyorder=1)
        y = np.clip(y, 0, 1)

    # then, use polyorder = 2 to smooth the shape
    wf = partial(window_func, max_window=150, dense_left=left, dense_right=right,
                 min_window=7)

    for i in range(5):
        y = variable_savgol_filter(y, wf, polyorder=2)
        y = np.clip(y, 0, 1)

    # ensure the monotone
    y = np.maximum.accumulate(y)

    return a, y


data.sample(1)

df = _f(data,
        country="TJK",
        year=2015,
        reporting_level="national")
# for a normal distubrition, 3.29 sigma captures 99.9% points.
pdf = create_pdf_and_remove_outliners(df, 3.3)
pdf_ = pdf.select(
    (1 - pl.col('headcount').sum()) *
    pl.col('headcount').map_batches(generate_weights) + pl.col('headcount')
)
cdf_clean = pdf_.select(
    pl.lit(0).append(pl.col('literal')).cum_sum().alias('headcount')
).with_row_index('i')

a, y_ = create_smooth_pdf_shape_(cdf_clean['headcount'].to_numpy())


plt.plot(np.diff(df["headcount"]))
plt.plot(fix_head(np.diff(y_), a))
# plt.plot(np.diff(y_))
plt.show()

plt.plot(df["i"], df["headcount"])
plt.plot(df["i"], y_)
plt.show()


d = y_ - df["headcount"]
plt.plot(d)
plt.vlines(200, d.min(), d.max(), color="blue", linestyles="dashed", alpha=0.5)
plt.show()
