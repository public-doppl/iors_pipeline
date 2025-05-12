import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, chi2
from sklearn.preprocessing import LabelEncoder

import utils.custom_statistics as Statistics

def get_pca_loadings_first_component(pca, columns):
    """
    Get the PCA loadings for the first component.
    Args:
        pca (PCA): The PCA object.
        columns (list): The list of feature names.
    Returns:
        pc1_loadings (pd.DataFrame): A DataFrame containing the feature names and their importance scores.
    """
    loadings = pd.DataFrame(
        data=pca.components_.T * np.sqrt(pca.explained_variance_), 
        columns=['Component 1', 'Component 2'],
        index=columns
    )
    pc1_loadings = loadings[['Component 1']].apply(lambda x: np.abs(x)) # take the absolute value of the loadings
    pc1_loadings = pc1_loadings.reset_index()
    pc1_loadings.columns = ['Feature', 'Importance']
    return pc1_loadings

def get_random_forest_features_importance(features, target):
    """
    Get the feature importance from a Random Forest model.
    Args:
        features (pd.DataFrame): The features DataFrame.
        target (pd.Series): The target Series.
    Returns:
        importances (pd.DataFrame): A DataFrame containing the feature names and their importance scores.
    """
    forest = RandomForestClassifier(random_state=0)
    forest.fit(features, target) # does not require normalization because not a distance-based model
    importances = pd.DataFrame({"Feature" : forest.feature_names_in_, 
                               "Importance" : forest.feature_importances_})
    return importances

def get_xgboost_features_importance(features, target):
    """
    Get the feature importance from an XGBoost model.
    Args:
        features (pd.DataFrame): The features DataFrame.
        target (pd.Series): The target Series.
    Returns:
        importances (pd.DataFrame): A DataFrame containing the feature names and their importance scores.
    """
    model = XGBClassifier()
    target_binary = LabelEncoder().fit_transform(target)  # Convert target to binary (0 and 1)
    model.fit(features, target_binary)  # does not require normalization because not a distance-based model
    importances = pd.DataFrame(data={
        'Feature': features.columns,
        'Importance': model.feature_importances_
    })
    return importances

def get_logistic_regression_features_importance(features, target):
    """
    Get the feature importance from a Logistic Regression model.
    Args:
        features (pd.DataFrame): The features DataFrame.
        target (pd.Series): The target Series.
    Returns:
        importances (pd.DataFrame): A DataFrame containing the feature names and their importance scores.
    """
    scaled_features = Statistics.get_standardized_data(features)

    model = LogisticRegression()
    model.fit(scaled_features, target)

    importances = pd.DataFrame(data={
        'Feature': features.columns,
        'Importance': np.abs(model.coef_[0]) # take the absolute value of the coefficients
    })
    return importances

def get_anova_f_scores(features, target):
    """
    Get the ANOVA F-scores for feature selection.
    Args:
        features (pd.DataFrame): The features DataFrame.
        target (pd.Series): The target Series.
    Returns:
        importances (pd.DataFrame): A DataFrame containing the feature names and their importance scores.
    """
    fs = SelectKBest(score_func=f_classif,k="all")

    target_numerical = target == 'Control'
    fit = fs.fit(features,target_numerical)

    importances = pd.DataFrame(data={
        'Feature': features.columns,
        'Importance': fit.scores_
    })
    return importances

def get_chi_squared_stats(features, target):
    """
    Get the Chi-squared statistics for feature selection.
    Args:
        features (pd.DataFrame): The features DataFrame.
        target (pd.Series): The target Series.
    Returns:
        importances (pd.DataFrame): A DataFrame containing the feature names and their importance scores.
    """
    fs = SelectKBest(score_func=chi2,k="all")

    target_numerical = target == 'Control'
    fit = fs.fit(features,target_numerical)

    importances = pd.DataFrame(data={
        'Feature': features.columns,
        'Importance': fit.scores_
    })
    return importances