import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from typing import List
import pandas as pd

def is_an_outlier(df, columns):
    """
    Check if the values in the specified columns are outliers based on the IQR method.
    Args:
        df (pd.DataFrame): The DataFrame to check for outliers.
        columns (list): List of columns to check for outliers.
    Returns:
        outliers: A boolean array indicating whether each value is an outlier.
    """
    outliers = np.full(len(df), False)
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        low_bound = Q1 - 1.5 * IQR
        high_bound = Q3 + 1.5 * IQR
        outliers[(df[column] < low_bound) | (df[column] > high_bound)] = True
    return outliers

def get_standardized_data(data):
    """
    Standardize the data using StandardScaler.
    Args:
        data (pd.DataFrame or array): The data to standardize.
    Returns:
        scaled_data: The standardized data.
    """
    # Standardize the Data
    scaler = StandardScaler()
    scaler.fit(data)
    scaled_data = scaler.transform(data)
    return scaled_data

def get_pca_transformed_data(df, n_components=2):
    """
    Perform PCA on the given DataFrame and return the PCA object and transformed data.
    Args:
        df (pd.DataFrame): The DataFrame to perform PCA on.
        n_components (int): The number of components to keep.
    Returns:
        pca: The PCA object.
        x_pca: The transformed data.
    """
    # Standardize the Data
    scaled_data = get_standardized_data(df)

    # PCA
    pca = PCA(n_components=n_components)
    pca.fit(scaled_data)
    x_pca = pca.transform(scaled_data)
    
    return pca, x_pca

def min_max_scaler(df):
    """
    Apply Min-Max scaling to the given DataFrame.
    Args:
        df (pd.DataFrame): The DataFrame to scale.
    Returns:
        scaled_data: The scaled data.
    """
    # Min-Max scaling
    min_max_scaler = MinMaxScaler()
    scaled_data = min_max_scaler.fit_transform(df)
    return scaled_data