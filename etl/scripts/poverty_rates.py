"""Calculate poverty rates for different poverty lines and country/regions"""

import os
import numpy as np
import polars as pl
import pandas as pd
import bracketlib
from functools import lru_cache


# ============================================================================
# Data Loading and Preparation
# ============================================================================


def load_base_data():
    """Load and prepare base data"""
    data = pl.read_parquet("../build/bridged_shapes.parquet")
    total_pop = pl.read_csv("../build/source/gapminder/population.csv")
    total_pop = total_pop.select(
        pl.col("geo").alias("country"),
        pl.col("time").alias("year").cast(pl.Int32),
        pl.col("Population").alias("population"),
    )
    data = data.join(total_pop, on=["country", "year"], how="left", suffix="_total")
    return data


def get_group_data(data, group_name):
    """Get aggregated data for a specific group type

    Args:
        data: Base data with country, year, bracket, population
        group_name: One of 'global', 'country', 'income_group', 'world_4region'

    Returns:
        Polars DataFrame aggregated by the appropriate dimensions
    """
    if group_name == "global":
        # Aggregate to global level and add 'global' column
        data_grouped = data.group_by(["year", "bracket"]).agg(pl.col("population").sum())
        data_grouped = data_grouped.select(
            pl.lit("world").alias("global"), pl.col("year"), pl.col("bracket"), pl.col("population")
        )
        return data_grouped

    elif group_name == "country":
        return data

    elif group_name == "income_group":
        # Load income group mappings
        wb_groups = pd.read_csv("../build/source/gapminder/wb_income_groups.csv")
        wb = wb_groups[["geo", "time", "4 income groups (WB)"]].copy()
        wb.columns = ["geo", "time", "level"]

        on_income = pd.read_csv("../build/source/gapminder/on_income_groups.csv")
        on_income = on_income[["geo", "time", "Income levels"]].dropna()
        on_income.columns = ["geo", "time", "level"]

        on_map = {
            "Level 1": "l1",
            "Level 2": "l2",
            "Level 3": "l3",
            "Level 4": "l4",
            "Level 5": "l4",
            "Level 6": "l4",
            "Level 7": "l4",
        }

        wb_map = {
            "Low income": "l1",
            "Lower middle income": "l2",
            "Upper middle income": "l3",
            "High income": "l4",
        }

        on = on_income.copy()
        on["level"] = on["level"].map(lambda x: on_map[x])
        wb["level"] = wb["level"].map(lambda x: wb_map[x])

        wb = wb.set_index(["geo", "time"])
        on = on.set_index(["geo", "time"])

        on = on.merge(wb, how="outer", on=["geo", "time"], suffixes=["_on", "_wb"])
        on["level"] = on["level_on"].fillna(on["level_wb"])
        on = on[["level"]]

        concept_id_map = {
            "l1": "low_income",
            "l2": "lower_middle_income",
            "l3": "upper_middle_income",
            "l4": "high_income",
        }

        income_groups = pl.DataFrame(on.reset_index()).select(
            pl.col("geo").alias("country"),
            pl.col("time").cast(pl.Int32).alias("year"),
            pl.col("level").replace_strict(concept_id_map),
        )

        data_income_level = (
            data.join(income_groups, on=["country", "year"], how="left")
            .group_by(pl.col(["level", "year", "bracket"]))
            .agg(pl.col("population").sum())
            .sort(["level", "year", "bracket"])
        )
        return data_income_level

    elif group_name == "world_4region":
        countries = pd.read_csv("../../ddf--entities--geo--country.csv")
        countries_regions_mapping = (
            countries.set_index("country")["world_4region"].dropna().to_dict()
        )

        data_world_4region = (
            data.select(
                pl.col("country").replace_strict(countries_regions_mapping).alias("world_4region"),
                pl.col("year"),
                pl.col("bracket"),
                pl.col("population"),
            )
            .group_by("world_4region", "year", "bracket")
            .agg(pl.col("population").sum())
        )
        return data_world_4region

    else:
        raise ValueError(f"Unknown group_name: {group_name}")


# ============================================================================
# Core Calculation Functions
# ============================================================================

step = bracketlib.get_bracket_step(500)


@lru_cache()
def income_from_bracket(x):
    """Cached function to convert bracket to income (log10 scale)"""
    return np.log10(bracketlib.income_from_bracket(x, step, integer=False))


def calculate_poverty_metric(df, povline, rates=True, above=False):
    """Calculate poverty metric for a single group

    Args:
        df: DataFrame with bracket and population columns for one group
        povline: Poverty line value (e.g., 3, 4.20, 200)
        rates: If True, return percentage rate; if False, return population count
        above: If False, calculate under the line; if True, calculate above the line

    Returns:
        Float (rate %) or Int (population count)
    """
    upper_bracket = bracketlib.bracket_from_income(povline, step)
    lower_bracket = upper_bracket - 1
    xloc = np.log10(povline)

    total_pop_data = df["population"].sum()

    df_ = (
        df.sort("bracket")
        .with_columns((pl.col("population").cum_sum() / total_pop_data).alias("poverty_pop"))
        .filter(pl.col("bracket").is_in([lower_bracket, upper_bracket]))
        .with_columns(pl.col("bracket").map_elements(income_from_bracket, return_dtype=pl.Float64))
    )

    if df_.is_empty():
        if rates:
            return 0.0
        return 0

    x = df_["bracket"].to_numpy()
    y = df_["poverty_pop"].to_numpy()

    xnew = np.array([xloc])
    ynew = np.interp(xnew, x, y)

    # Handle above/below toggle
    result = ynew[0]
    if above:
        result = 1 - result

    if not rates:
        # Convert to population count
        result = int(result * total_pop_data)
    else:
        # Convert to percentage
        result = result * 100

    return result


def get_poverty_for_groups(df, by, povline, rates=True, above=False, indicator_name=None):
    """Calculate poverty metrics for all groups in the data

    Args:
        df: Full DataFrame with all data
        by: List of column names to group by (e.g., ['year'], ['country', 'year'])
        povline: Poverty line value
        rates: If True, return rates; if False, return counts
        above: If False, under the line; if True, above the line
        indicator_name: Optional custom indicator name; if None, auto-generate

    Returns:
        Polars DataFrame with group columns + indicator column
    """
    if indicator_name is None:
        indicator_name = generate_indicator_name(povline, rates, above)

    datalist = df.partition_by(by, as_dict=True)
    res = list()

    for k, _df in datalist.items():
        res_part = dict(zip(by, k))
        res_part[indicator_name] = calculate_poverty_metric(
            _df, povline=povline, rates=rates, above=above
        )
        res.append(res_part)

    return pl.from_records(res)


# ============================================================================
# Naming and Saving Utilities
# ============================================================================


def generate_indicator_name(povline, rates=True, above=False):
    """Generate consistent indicator names

    Examples:
        (3, True, False) -> "poverty_rate_under_3"
        (3, False, False) -> "population_under_3"
        (200, False, True) -> "population_above_200"
        (4.20, True, False) -> "poverty_rate_under_4_20"
    """
    # Format poverty line (replace . with _)
    pov_str = str(povline).replace(".", "_")

    # Determine metric type
    metric_type = "poverty_rate" if rates else "population"

    # Determine direction
    direction = "above" if above else "under"

    return f"{metric_type}_{direction}_{pov_str}"


def save_poverty_data(df, by_dimensions, indicator_name, output_dir="./ddf/poverty_rates/"):
    """Save poverty data with DDF-compliant naming

    Args:
        df: DataFrame to save
        by_dimensions: List of dimension column names (e.g., ['country', 'time'])
        indicator_name: The indicator name (matches column header)
        output_dir: Output directory

    Generates filename: ddf--datapoints--{indicator}--by--{dim1}--{dim2}.csv
    """
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename
    by_str = "--".join(by_dimensions)
    filename = f"ddf--datapoints--{indicator_name}--by--{by_str}.csv"
    filepath = os.path.join(output_dir, filename)

    # Save
    df.write_csv(filepath)
    print(f"Saved: {filepath}")


# ============================================================================
# Main Function
# ============================================================================


def main():
    """Main function to calculate and save poverty rates"""

    # Load base data
    print("Loading data...")
    data = load_base_data()

    # Configuration: (group_name, by_columns, povline, rates, above)
    configs = [
        # Global - rates under poverty lines
        ("global", ["global", "year"], 3, True, False),
        ("global", ["global", "year"], 12, True, False),
        ("global", ["global", "year"], 48, True, False),
        ("global", ["global", "year"], 48, True, True),  # above 48
        # Country - population counts
        ("country", ["country", "year"], 3, False, False),
        # # More Global - population counts (commented)
        # ('global', ['global', 'year'], 3, False, False),
        # ('global', ['global', 'year'], 2, False, False),
        # ('global', ['global', 'year'], 200, False, True),
        # # Country - rates under poverty lines
        # ('country', ['country', 'year'], 3, True, False),
        # ('country', ['country', 'year'], 4.20, True, False),
        # ('country', ['country', 'year'], 8.30, True, False),
        # # Country - population counts
        # ('country', ['country', 'year'], 3, False, False),
        # ('country', ['country', 'year'], 2, False, False),
        # ('country', ['country', 'year'], 200, False, True),
        # # Income groups - rates
        # ('income_group', ['level', 'year'], 3, True, False),
        # ('income_group', ['level', 'year'], 4.20, True, False),
        # ('income_group', ['level', 'year'], 8.30, True, False),
        # # World 4 regions - rates
        # ('world_4region', ['world_4region', 'year'], 3, True, False),
        # ('world_4region', ['world_4region', 'year'], 4.20, True, False),
        # ('world_4region', ['world_4region', 'year'], 8.30, True, False),
    ]

    # Process each configuration
    for config in configs:
        group_name, by, povline, rates, above = config

        print(f"\nProcessing: {group_name}, povline={povline}, rates={rates}, above={above}")

        # Get appropriate data subset
        df = get_group_data(data, group_name)

        # Generate indicator name
        indicator = generate_indicator_name(povline, rates, above)

        # Calculate metrics
        result = get_poverty_for_groups(df, by, povline, rates, above, indicator)

        # Rename 'year' to 'time' if present
        if "year" in result.columns:
            result = result.rename({"year": "time"})

        # Update by_dimensions for saving
        final_by = ["time" if col == "year" else col for col in by]

        # Sort for consistency
        result = result.sort(final_by)

        # Save
        save_poverty_data(result, final_by, indicator)


if __name__ == "__main__":
    main()
