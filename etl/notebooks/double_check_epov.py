import numpy as np
import polars as pl
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from io import BytesIO


country_epov_rates = pl.read_csv(
    "../scripts/ddf/poverty_rates/ddf--datapoints--extereme_poverty_rate--by--country--time.csv"
)

country_epov_rates


def download(url, num_retries=3):
    # Set up a retry strategy in case of connection issues
    retry_strategy = Retry(
        total=num_retries,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
        backoff_factor=3,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    http = requests.Session()
    http.mount("https://", adapter)
    http.mount("http://", adapter)

    file_to_download = BytesIO()
    with http.get(url, stream=True) as response:
        response.raise_for_status()  # Check for HTTP issues
        for chunk in response.iter_content(chunk_size=8192):
            file_to_download.write(chunk)

    print("Downloaded successfully.")
    file_to_download.seek(0)
    return file_to_download


url_tmpl = "https://api.worldbank.org/pip/v1/pip?country=all&year=all&povline={}&fill_gaps=true&group_by=none&welfare_type=all&reporting_level=all&format=csv&ppp_version=2017"


f = download(url_tmpl.format(2.15))

wb_data = pl.read_csv(f)

wb_data

wb_data.columns

wb_data.select(["country_code", "reporting_year", "reporting_level", "headcount"])

levels = {"national": 0, "urban": 1, "rural": 2}

filtered = wb_data.select(
    pl.col("country_code").str.to_lowercase().alias("country"),
    pl.col("reporting_year").alias("year"),
    pl.col("reporting_level").replace_strict(levels),
    pl.col("headcount").alias("rate"),
)

filtered

wb_epov = (
    filtered.group_by("country", "year")
    .agg(
        pl.col("rate").filter(
            pl.col("reporting_level") == pl.col("reporting_level").min()
        )
    )
    .with_columns(pl.col("rate").list.first() * 100)
    .sort(["country", "year"])
    .with_columns(pl.col("country").replace({"xkx": "kos"}))
)

wb_epov

gm_epov = country_epov_rates.select(
    pl.col("country"),
    pl.col("time").alias("year"),
    pl.col("extreme_poverty_rate").alias("gm_rates"),
)

diff = gm_epov.join(wb_epov, on=["country", "year"], how="right")

diff.select(
    pl.col(["country", "year", "gm_rates", "rate"]),
    np.abs(pl.col("rate") - pl.col("gm_rates")).alias("diff"),
).write_csv("./gm_rates_wb_rates_diff.csv")

# Let's try to analysis big differences
# possible cause 1:
# - there are lots of noise in povcalnet data
import sys

sys.path.append("../scripts")

import bracketlib

bracketlib.bracket_from_income(2.15, 0.04)

wb_data1 = pl.read_csv("../build/source/povcalnet/0202.csv")
wb_data1.columns

wb_data1.select(
    pl.col(
        [
            "country_code",
            "reporting_year",
            "reporting_level",
            "poverty_line",
            "headcount",
        ]
    )
).filter(pl.col("country_code") == "COL")

wb_epov.filter(pl.col("country") == "col")


wb_data2 = pl.read_csv("../build/source/povcalnet/0203.csv")

wb_data2.select(
    pl.col(
        [
            "country_code",
            "reporting_year",
            "reporting_level",
            "poverty_line",
            "headcount",
        ]
    )
).filter(pl.col("country_code") == "COL")


# create a dataframe for given country, year
def _f(df, **kwargs):
    return df.filter(pl.all_horizontal([(pl.col(k) == v) for k, v in kwargs.items()]))


def get_data(n, country_code, year, reporting_level):
    path = f"../build/source/povcalnet/0{n}.csv"
    data = pl.read_csv(path)
    return _f(
        data.select(
            pl.col(
                [
                    "country_code",
                    "reporting_year",
                    "reporting_level",
                    "poverty_line",
                    "headcount",
                ]
            )
        ),
        country_code=country_code,
        reporting_year=year,
        reporting_level=reporting_level,
    )


df_tocheck = pl.concat(
    [get_data(i, "BDI", 2001, "national") for i in range(130, 230)],
    how="vertical_relaxed",
)

df_tocheck

import matplotlib.pyplot as plt

xs = np.log2(df_tocheck["poverty_line"])
ys = df_tocheck["headcount"].to_numpy()

plt.plot(xs, ys)
plt.show()

# that seems normal! so it must something in smoothing and bridging.
smoothed = pl.read_parquet("../build/povcalnet_smoothed.parquet")
smoothed

_f(smoothed, country="bdi", year=2001)

df_tocheck2 = (
    _f(smoothed, country="bdi", year=2001, reporting_level="n")
    .filter(pl.col("bracket").is_between(130, 230))
    .select(
        # pl.col('bracket'),
        pl.col("bracket")
        .map_elements(
            lambda x: bracketlib.income_from_bracket(
                x, 0.04, bound="upper", integer=False
            )
        )
        .alias("poverty_line"),
        pl.col("headcount").cum_sum(),
    )
)

df_tocheck2
df_tocheck


xs = np.log2(df_tocheck["poverty_line"])
ys = df_tocheck["headcount"].to_numpy()
plt.plot(xs, ys)

xs = np.log2(df_tocheck2["poverty_line"])
ys = df_tocheck2["headcount"].to_numpy()
plt.plot(xs, ys)


plt.show()
