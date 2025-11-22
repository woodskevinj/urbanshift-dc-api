import pandas as pd
import numpy as np
import geopandas as gpd

def compute_crime_rate(df):
    
    """
    Compute crime rate per 1000 residents.
    Requires:
    - total_crime
    - population
    """
    df = df.copy()
    df["crime_rate_per_1000"] = (
        df["total_crime"] / df["population"] * 1000
    )
    return df

def compute_uplift_score(df, w1=0.6, w2=0.3, w3=0.1):
    """
    Compute uplift potential score using weighted factors.
    Current factors:
    - w1 -> crime_rate_per_1000 (inverse relationship)
    - w2 -> accessibility_score (Direct)
    - w3 -> home_value_score (Direct)
    Note:  Will engineer accessibility_score and home_value_score later.
    For now, this function expects them to already be present in df.
    """
    df = df.copy()

    # To avoid division-by-zero or NaN issues:
    df = df.fillna(0)

    # Invert crim (lower crime = higher uplift)
    df["crime_inverse"] = 1 / (1 + df["crime_rate_per_1000"])

    df["uplift_score"] = (
        w1 * df["crime_inverse"]
        + w2 * df["accessibility_score"]
        + w3 * df["home_value_score"]
    )

    return df

def prepare_model_features(df):
    """
    Prepare final features for TensorFlow modeling.
    Returns clean numeric X matrix and y target.

    Expected columns:
    - uplift_score (target)
    - crime_rate_per_1000
    - accessibility_score
    - home_value_score
    """
    df = df.copy()

    feature_cols = [
        "crime_rate_per_1000",
        "accessibility_score",
        "home_value_score",
    ]

    X = df[feature_cols].values
    y = df["uplift_score"].values

    return X, y


def compute_home_value_trend(home_value_df: pd.DataFrame,
                             years: list[int]
                             ) -> pd.DataFrame:
    """
    Returns DataFrame:
    - tract_id, home_value (% change over last N years)
    """
    pass

def compute_amenity_gradient(store_points: gpd.GeoDataFrame,
                             tract_geoms: gpd.GeoDataFrame
                             ) -> pd.DataFrame:
    """
    For each tract: compute density of stores within 0-.5mi vs .5-1mi.
    Return: tract_id, amenity_gradient
    """
    pass
