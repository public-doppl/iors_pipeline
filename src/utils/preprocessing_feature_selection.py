import utils.config as config
import utils.custom_statistics as Statistics

def filter_cytokine_timepoints(df, cytokine_days):
    """
    Filters the DataFrame to include only the specified days where cytokines effect is the strongest.
    Args:
        df (pd.DataFrame): The DataFrame to filter.
        cytokine_days (list): List of days to consider.
    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    return df[df['Day'].isin(cytokine_days)]

def filter_conditions_of_interest(df, target_conditions_mapping):
    """
    Filters the DataFrame to include only the specified conditions of interest and maps them to a target condition.
    Args:
        df (pd.DataFrame): The DataFrame to filter.
        conditions_of_interest (list): List of conditions to consider.
    Returns:
        pd.DataFrame: The filtered DataFrame.
    """ 
    filtered_df = df[df['Condition'].isin(target_conditions_mapping.keys())]
    filtered_df.loc[:,'Condition'] = filtered_df['Condition'].map(target_conditions_mapping)
    return filtered_df

def discard_outliers(features, target, outlier_features=[]):
    """
    Discards outliers from the features and target DataFrames based on the specified outlier features.
    Args:
        features (pd.DataFrame): The features DataFrame.
        target (pd.Series): The target Series.
        outlier_features (list): List of features to check for outliers.
    Returns:
        tuple: A tuple containing the features and target DataFrames without outliers.
    """
    outliers = Statistics.is_an_outlier(features, outlier_features)
    features_no_outliers = features[~ outliers]
    target_no_outliers = target[~ outliers]

    is_na = features_no_outliers.isna().any(axis=1)
    features_no_outliers = features_no_outliers[~ is_na]
    target_no_outliers = target_no_outliers[~ is_na]
    return features_no_outliers, target_no_outliers

def preprocess_dataset(raw_dataset, cytokine_days, outlier_features=[]):
    """
    Preprocesses the dataset by filtering cytokine timepoints, conditions of interest, and discarding outliers.
    Args:
        raw_dataset (pd.DataFrame): The raw dataset to preprocess.
        cytokine_days (list): List of days to consider for cytokine effect.
    Returns:
        tuple: A tuple containing the preprocessed features and target DataFrames.
    """
    # Filter the dataset to include only the specified days
    filtered_dataset = filter_cytokine_timepoints(raw_dataset, cytokine_days)

    # Filter the dataset to include only the specified conditions of interest
    filtered_dataset = filter_conditions_of_interest(filtered_dataset, config.FEATURE_SELECTION_CONDITIONS_MAPPING)

    # Discard outliers from the features and target DataFrames
    if len(outlier_features) > 0:
        features_no_outliers, target_no_outliers = discard_outliers(
            filtered_dataset.drop(columns=['Condition', 'Well', 'Day', 'Organoid_index']),
            filtered_dataset['Condition'],
            outlier_features
        )
    else:
        features_no_outliers = filtered_dataset.drop(columns=['Condition', 'Well', 'Day', 'Organoid_index'])
        target_no_outliers = filtered_dataset['Condition']

    return features_no_outliers, target_no_outliers



