import numpy as np

import utils.config as config

def pivot_dataset(df):
    """
    Pivot the dataset to have days as columns and compute mean values for each organoid.
    Args:
        df (pd.DataFrame): The DataFrame to pivot.
    Returns:
        pd.DataFrame: The pivoted DataFrame with mean values for each organoid.
    """
    ALL_COLUMNS_OF_INTEREST = config.IORS_HELPER_COLUMNS + config.IORS_FEATURES_OF_INTEREST
    df_filtered = df[ALL_COLUMNS_OF_INTEREST].copy()
    df_filtered = df_filtered.groupby(config.IORS_HELPER_COLUMNS).agg({'Area (um2)': 'mean', 'Normalized mean intensity':'mean'}).reset_index()
    df_pivot = df_filtered.pivot(index=[col for col in config.IORS_HELPER_COLUMNS if col != 'Day'], columns='Day', values=config.IORS_FEATURES_OF_INTEREST)
    return df_pivot

def remove_organoids_with_missing_timepoints(pivot_df):
    """
    Remove organoids with missing timepoints in the pivoted DataFrame.
    Args:
        pivot_df (pd.DataFrame): The pivoted DataFrame.
    Returns:
        pd.DataFrame: The filtered DataFrame with organoids having all timepoints.
    """
    initial_length = len(pivot_df)
    # If missing intensity value, missing area value too
    intensity_columns = [col for col in pivot_df.columns if 'intensity' in col[0]]
    new_pivot_df = pivot_df[pivot_df[intensity_columns].isna().sum(axis=1) == 0]
    return new_pivot_df

def normalize_pivot_dataset(df_pivot):
    """
    Normalize the pivoted dataset by dividing each value by the corresponding T0 value.
    Args:
        df_pivot (pd.DataFrame): The pivoted DataFrame to normalize.
    Returns:
        pd.DataFrame: The normalized DataFrame.
    """
    df_pivot_normalized = df_pivot.copy()
    days = np.unique([col[1] for col in df_pivot.columns])
    reference_day = 'T0'
    if reference_day not in days:
        raise ValueError(f"Reference day {reference_day} not found in the dataset. Available days: {days}")
    # Normalize values by dividing by T0 values
    for day in days:
        df_pivot_normalized[('Area (um2)', day)] = df_pivot[('Area (um2)', day)] / df_pivot[('Area (um2)', reference_day)]
        df_pivot_normalized[('Normalized mean intensity', day)] = df_pivot[('Normalized mean intensity', day)] / df_pivot[('Normalized mean intensity', reference_day)]
    return df_pivot_normalized

def preprocess_dataset(df):
    """
    Preprocess the dataset by filtering for specific days, pivoting, normalizing, and removing organoids with missing timepoints.
    Args:
        df (pd.DataFrame): The DataFrame to preprocess.
    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    filtered_df = df.copy()
    # Filter the dataset to include only the days of interest
    filtered_df = filtered_df[filtered_df['Day'].isin(config.IORS_DAYS_OF_INTEREST)]
    # Pivot the dataset
    pivot_df = pivot_dataset(filtered_df)
    # Normalize the dataset
    pivot_df = normalize_pivot_dataset(pivot_df)
    final_df = remove_organoids_with_missing_timepoints(pivot_df)
    return final_df