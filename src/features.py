import pandas as pd
import numpy as np
import geopandas as gpd

def compute_crime_rate(crime_df: pd.DataFrame,
                       arrests_df: pd.DataFrame,
                       population_df: pd.DataFrame
                       ) -> pd.DataFrame:
    
    """
    Returns a DataFrame with columns:
    - tract_id, year, crime_rate (per 1000)
    """
    # steps: filter violent & drug arrest, group by tract/year,
    # divide by population, *1000
    pass

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
