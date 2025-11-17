import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Compute features: crime rate, home value trend, amenity gradient.

def compute_z_scores(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Adds columns: e.g. Z_crime_rate, Z_home_value_trend, Z_amenity_gradient
    """
    scaler = StandardScaler()
    z_array = scaler.fit_transform(df[feature_cols])
    z_df = pd.DataFrame(z_array, columns=[f"Z_{c}" for c in feature_cols], index=df.index)
    return pd.concat([df.reset_index(drop=True), z_df], axis=1)

def compute_uplift_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies formula:
        UpliftScore = 0.4*Z_crime_rate + 0.4*Z_neg_home_value_trend + 0.2*Z_amenity_gradient
        (or extended version if you include transit)
    """
    df = df.copy()
    df["Z_neg_home_value_trend"] = -1 * df["Z_home_value_trend"]
    df["UpliftScore"] = (
        0.4*df["Z_crime_rate"] +
        0.4*df["Z_neg_home_value_trend"] +
        0.2*df["Z_amenity_gradient"]
    )
    return df
