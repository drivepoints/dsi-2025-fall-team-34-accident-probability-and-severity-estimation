import pandas as pd
import numpy as np

def get_cyclical_features(df, col_name, max_val):
    """Encodes a time column into Sin/Cos."""
    df[f'{col_name}_sin'] = np.sin(2 * np.pi * df[col_name] / max_val)
    df[f'{col_name}_cos'] = np.cos(2 * np.pi * df[col_name] / max_val)
    return df

def engineer_features(df):
    """
    Applies all transformations. 
    Expects input cols: ['hour', 'day_of_week', 'month']
    """
    df = df.copy()

    df = get_cyclical_features(df, 'hour', 24)
    df = get_cyclical_features(df, 'day_of_week', 7)
    df = get_cyclical_features(df, 'month', 12)

    return df
