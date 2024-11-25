import numpy as np
import pandas as pd
import polars as pl
from scipy import interpolate
from scipy.optimize import minimize
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from smoothlib import run_smooth
from kalmangrad import grad


# let's import the cleaned povcalnet data
data = pl.read_parquet("../build/povcalnet_clean.parquet")
data


def _f(df, **kwargs):
    return df.filter(pl.all_horizontal([(pl.col(k) == v) for k, v in kwargs.items()]))


# regulate points
# def regulate_points(y, min_diff=1e-7, max_diff=10):
#     res_y = [0]

#     for i in range(1, len(y)):
#         y_t_ = res_y[-1]
#         y_measure = y[i]

#         if y_t_ == 0:
#             if y_measure > min_diff:
#                 res_y.append(min_diff)
#             else:
#                 res_y.append(y_measure)
#         else:
#             if np.abs(y_measure - y_t_) < min_diff:
#                 res_y.append(y_t_ + min_diff)
#             else:
#                 if np.ln(y_measure / y_t_) > max_diff:
#                     res_y.append(y_t_ + max_diff)
#                 elif (y_measure - y_t_) < - max_diff:
#                     res_y.append(y_t_ - max_diff)
#                 else:
#                     res_y.append(y_measure)

#     return res_y


# 1. just get the PDF and remove some outliners


def remove_outliers(x, y, n_mean=10, window_size=21):
    y_ = pd.Series(y)
    y_mean = y_.rolling(window=window_size, center=True, min_periods=1).mean()
    # print(y_mean)
    thres = n_mean * y_mean

    mask = y_ <= thres

    return x[mask], y[mask]


def get_noisy_pdf(x, y):
    y = np.diff(y)

    # back fill the first value
    y = np.insert(y, 0, y[0])

    assert x.shape == y.shape

    # return remove_outliers(x, y)
    return x, y


# example data
# CHN,2000-2005 issue
example = _f(data, country="NOR", year=2000, reporting_level="national")
example

xs = example["i"]
ys = example["headcount"]

yp = regulate_points(ys, max_diff=1e-2)
yp

plt.figure(figsize=(10, 6))
plt.scatter(xs, ys, label="Original Data", alpha=0.5)
plt.plot(xs, yp, "r-", label="Regulated CDF")
plt.xlabel("X")
plt.ylabel("Cumulative Probability")
plt.title("Smoothed Cumulative Distribution Function")
plt.legend()
plt.grid(True)
plt.show()


xp, yp = get_noisy_pdf(xs.to_numpy(), regulate_points(ys.to_numpy(), max_diff=3))

yp.sum()

smoother_states, filter_times = grad(yp, xp, n=2, delta_t=1, obs_noise_std=0.01)
len(filter_times)
yp_ = [s.mean()[0] for s in smoother_states]
sum(yp_)

plt.scatter(xs[1:], np.diff(ys), color="r", alpha=0.5, s=5)
plt.plot(filter_times, yp_)
plt.show()


def preprocess_data(x, y):
    # Record min and max values
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)

    # Remove consecutive points with the same value
    diff = np.diff(y, prepend=0)
    mask = (np.abs(diff) > 1e-4) & (diff > 0)
    x, y = x[mask], y[mask]

    # Append or set min_x to min_y and max_x to max_y
    if x[0] != min_x:
        x = np.insert(x, 0, min_x)
        y = np.insert(y, 0, min_y)
    else:
        y[0] = min_y

    if x[-1] != max_x:
        x = np.append(x, max_x)
        y = np.append(y, max_y)
    else:
        y[-1] = max_y

    # also extend right side
    x = np.append(x, 500)
    y = np.append(y, 1)

    return x, y


def estimate_noise_level(y):
    # Estimate noise using the difference between adjacent points
    diff = np.diff(y)
    noise_level = np.std(diff)
    return noise_level


def adaptive_window_length(noise_level, n_points):
    # Adjust window length based on noise level and number of points
    # Start with 10% of data points, minimum 3
    base_window = max(int(n_points * 0.1), 3)
    noise_factor = int(noise_level * 100)  # Scale noise to usable range

    window = base_window + noise_factor
    # Ensure window is smaller than data
    window = min(window, (n_points - 1) // 2)
    window = window if window % 2 == 1 else window + 1  # Ensure odd number

    return window


def smooth_and_monotonize_cdf(x, y, noise_level=None):
    # Preprocess the data
    x_processed, y_processed = preprocess_data(x, y)

    if not noise_level:
        noise_level = estimate_noise_level(y_processed)

    n_points = len(x_processed)
    window_length = adaptive_window_length(noise_level, n_points)
    polyorder = min(3, window_length - 1)  # Adjust polyorder if necessary

    print(f"Preprocessed data points: {n_points}")
    print(f"Estimated noise level: {noise_level:.4f}")
    print(f"Adaptive window length: {window_length}")
    print(f"Polynomial order: {polyorder}")

    # Step 1: Smooth the data using Savitzky-Golay filter
    y_smoothed = savgol_filter(y_processed, window_length, polyorder)

    # Step 2: Ensure monotonicity
    y_monotone = np.maximum.accumulate(y_smoothed)

    # Step 3: Clip values to [0, 1] range
    y_monotone = np.clip(y_monotone, 0, 1)

    # Step 3: Create a monotonic interpolation
    # f = interpolate.PchipInterpolator(x_processed, y_monotone)
    f = interpolate.interp1d(
        x_processed, y_monotone, kind="linear", bounds_error=False, fill_value=(0, 1)
    )

    return f


# NEW METHOD:
# use kalmangrad to estimate the CDF and PDF all at once.


xp, yp = preprocess_data(xs.to_numpy(), ys.to_numpy())
len(yp)


# NEW:
estimate_noise_level(yp)
smoother_states, filter_times = grad(yp, xp, n=2, delta_t=1, obs_noise_std=0.03)
smoothed_cdf = np.array([state.mean()[0] for state in smoother_states])
# smoothed_cdf = np.clip(smoothed_cdf, 0, 1)
# smoothed_cdf = np.maximum.accumulate(smoothed_cdf)
len(smoothed_cdf)
filter_times


# Create the smoothed and monotonic CDF
smoothed_cdf = smooth_and_monotonize_cdf(xs.to_numpy(), ys.to_numpy(), noise_level=0.03)

# Generate points for plotting
x_plot = np.arange(0, 500)
y_plot = smoothed_cdf(x_plot)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(xs, ys, label="Original Data", alpha=0.5)
plt.plot(x_plot, y_plot, "r-", label="Smoothed CDF")
# plt.plot(filter_times, smoothed_cdf, 'r--', label="smoothed cdf")
plt.xlabel("X")
plt.ylabel("Cumulative Probability")
plt.title("Smoothed Cumulative Distribution Function")
plt.legend()
plt.grid(True)
plt.show()

y_plot

#
smoother_states, filter_times = grad(y_plot, x_plot, n=2, delta_t=1, obs_noise_std=0.01)


estimated_pdf = [state.mean()[1] for state in smoother_states]
estimated_pdf
plt.plot(xs[1:], np.diff(ys), "r.", label="observed data")
# plt.plot(filter_times, estimated_pdf, 'g-', label='Smoothed 2')
# plt.plot(filter_times[1:], run_smooth_and_fix_area(np.diff(smoothed_cdf), 50, 0), 'g-', label='smoothed pdf')
plt.plot(filter_times, estimated_pdf, "g-", label="smoothed pdf 1", alpha=1)
plt.legend()
plt.grid(True)
plt.show()


# Generate points for plotting
x_plot = xs
y_plot = np.diff(smoothed_cdf(x_plot))
plt.plot(x_plot[1:], y_plot, "r-", label="Smoothed CDF")
plt.legend()
plt.grid(True)
plt.show()

# now we want to smooth the pdf


def smooth_pdf(x, y, smoothness=1.0, max_iterations=100, constraint_interval=5):
    """
    Smooth the input PDF while preserving a single maximum at the midpoint of observed maxima,
    and ensuring strict monotonicity before and after the maximum with reduced constraints.
    """
    # Find all indices of the maximum value
    max_value = np.max(y)
    max_indices = np.where(y == max_value)[0]

    area = np.sum(y)
    print(area)

    # Calculate the midpoint index of the maximum values
    mid_max_index = int(np.mean(max_indices))

    # Define the objective function to minimize
    def objective(y_smooth):
        # Smoothness term
        smoothness_term = np.mean(np.diff(y_smooth, 2) ** 2)
        # Fit term
        fit_term = np.mean((y - y_smooth) ** 2)
        return smoothness * smoothness_term + fit_term

    # Define constraints
    constraints = [
        # Ensure the total area under the curve remains 1
        {"type": "eq", "fun": lambda y_smooth: np.sum(y_smooth) - area},
        # Preserve the maximum value at the midpoint
        {"type": "eq", "fun": lambda y_smooth: y_smooth[mid_max_index] - max_value},
        # Ensure all y values are non-negative
        {"type": "ineq", "fun": lambda y_smooth: y_smooth},
    ]

    # Add reduced number of monotonicity constraints
    for i in range(0, mid_max_index, constraint_interval):
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda y_smooth, i=i: y_smooth[i + constraint_interval]
                - y_smooth[i]
                - 1e-10,
            }
        )

    for i in range(mid_max_index, len(y) - constraint_interval, constraint_interval):
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda y_smooth, i=i: y_smooth[i]
                - y_smooth[i + constraint_interval]
                - 1e-10,
            }
        )

    # Minimize the objective function
    result = minimize(
        objective,
        y,
        method="SLSQP",
        constraints=constraints,
        options={"maxiter": max_iterations},
    )

    return result.x


data.sample(1)
len(data.partition_by(["country", "year", "reporting_level"]))

example = _f(data, country="SLE", year=1981, reporting_level="national")
xs = example["i"]
ys = example["headcount"]
smoothed_cdf = smooth_and_monotonize_cdf(xs.to_numpy(), ys.to_numpy())
x_plot = xs
y_plot = np.diff(smoothed_cdf(x_plot))

# for _ in range(2):
#     y_plot_ = run_smooth(y_plot, 20, 0)
#     y_plot_ = y_plot_ / y_plot_.sum()

smoothed_y = smooth_pdf(
    x_plot[1:], y_plot, smoothness=50, max_iterations=5, constraint_interval=1
)

for _ in range(2):
    s0 = np.sum(smoothed_y)
    smoothed_y = run_smooth(smoothed_y, 20, 1)
    s1 = np.sum(smoothed_y)
    correction_factor = s0 / s1
    smoothed_y = correction_factor * smoothed_y

# for _ in range(2):
#     smoothed_y = run_smooth(smoothed_y, 10, 0)
#     smoothed_y = smoothed_y / smoothed_y.sum()

plt.plot(xs[1:], np.diff(ys), "r-", label="orig")
# plt.plot(x_plot[1:], y_plot_, 'b-', label='Smoothed 1')
plt.plot(x_plot[1:], smoothed_y, "g-", label="Smoothed 2")
plt.legend()
plt.grid(True)
plt.show()


smoothed_y[0]
ys[0]
ys.sum()
smoothed_y.sum()

np.cumsum(smoothed_y)

# calculate back the CDF from PDF.
# now I see we must add back the missing values so that the sum becomes 1
plt.plot(x_plot[:-1], ys[0] + np.cumsum(smoothed_y))
plt.scatter([202.608], [0.6416])
plt.show()


def run_smooth_and_fix_area(y, w, times=1):
    s0 = np.sum(y)
    smoothed_y = y
    for _ in range(times):
        smoothed_y = run_smooth(smoothed_y, w, 0)
    s1 = np.sum(smoothed_y)
    correction_factor = s0 / s1
    return correction_factor * smoothed_y


def func(x):
    """function to smooth a series"""
    # run smoothing, based on standard deviation
    std = x.std()
    s0 = np.sum(x)
    if std < 0.004:
        res = run_smooth(x, 30, 7)
        res = run_smooth(res, 30, 3)
    elif std <= 0.0045 and std > 0.004:
        res = run_smooth(x, 30, 5)
        res = run_smooth(res, 30, 2)
    elif std <= 0.0049 and std > 0.0045:
        res = run_smooth(x, 30, 3)
        res = run_smooth(res, 20, 2)
    elif std > 0.0049:
        res = run_smooth(x, 30, 2)
        res = run_smooth(res, 20, 1)
    s1 = np.sum(res)
    correction_factor = s0 / s1
    return correction_factor * res


# NEXT: let's do all shapes


def create_smoothed_shape(df):
    xs = df["i"]
    ys = df["headcount"]
    smoothed_cdf = smooth_and_monotonize_cdf(xs.to_numpy(), ys.to_numpy())

    xs_ = np.arange(0, 501, 1)
    smoothed_ys = smoothed_cdf(xs_)

    shape_xs = xs_[:-1]  # 0 - 500
    shape_ys = np.diff(smoothed_ys)

    smoothed_y = smooth_pdf(
        shape_xs, shape_ys, smoothness=50, max_iterations=5, constraint_interval=5
    )

    # print(np.std(smoothed_y))
    smoothed_y = func(smoothed_y)

    # for _ in range(2):
    #     smoothed_y = run_smooth_and_fix_area(smoothed_y, 30, 2)

    # for _ in range(2):
    #     smoothed_y = run_smooth_and_fix_area(smoothed_y, 20, 0)

    country = df["country"][0]
    year = df["year"][0]
    level = df["reporting_level"][0]
    return pl.DataFrame({"i": shape_xs, "headcount": smoothed_y}).select(
        pl.lit(country).alias("country"),
        pl.lit(year).alias("year"),
        pl.lit(level).alias("reporting_level"),
        pl.col(["i", "headcount"]),
    )


old_df = pl.read_parquet("../build/povcalnet_smoothed.parquet")

example = _f(data, country="IND", year=2016, reporting_level="national")
new_df = create_smoothed_shape(example)


odf = _f(old_df, country="ind", year=2016, reporting_level="n")

xs = example["i"]
ys = example["headcount"]
plt.plot(xs[:-1], np.diff(ys), "r-", label="orig", alpha=0.3)
plt.plot(odf["bracket"].to_numpy(), odf["headcount"].to_numpy(), label="old")
plt.plot(new_df["i"].to_numpy(), new_df["headcount"].to_numpy(), label="new")
plt.legend()
plt.grid()
plt.show()

old_df["bracket"].unique()

data.filter(
    pl.col("i") == 459,
).filter(pl.col("headcount").is_between(pl.col("headcount").min(), 0.9997))
