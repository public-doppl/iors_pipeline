import os
from functools import reduce
import pandas as pd
from pathlib import Path
import argparse
import logging

import utils.config as config
import utils.custom_io as IO
import utils.preprocessing_feature_selection as Preprocessing
import utils.visualization as Visualization
import utils.custom_statistics as Statistics
import utils.feature_importance_methods as FeatureImportance

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Feature selection pipeline",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_folder", help="input folder", required=True)
    parser.add_argument("--input_spreadsheet", help="input spreadsheet filename", required=True)
    parser.add_argument("--output_folder", help="output folder where stacks should be stored", required=True)
    parser.add_argument("--cytokine_days", help="list of days where cytokines effect is the strongest", default=config.FEATURE_SELECTION_DEFAULT_CYTOKINE_DAYS, nargs='*')
    parser.add_argument("--outlier_conditions", help="list of conditions that should be considered when discarding outliers", default=config.FEATURE_SELECTION_FEATURES_OF_INTEREST, nargs='*')
    args = parser.parse_args()

    ### Loading the dataset
    raw_dataset = IO.load_dataset(args.input_folder, args.input_spreadsheet)

    ### Creating output folders
    visualization_folder, spreadsheets_folder = IO.create_output_folders(args.output_folder)

    ### Preprocessing the dataset
    features, target = Preprocessing.preprocess_dataset(raw_dataset, args.cytokine_days, args.outlier_conditions)

    ### Explore inter-features correlation
    correlations = Visualization.plot_correlation_matrix(features, output_folder=visualization_folder, output_filename=f"{args.input_spreadsheet.split('.')[0]}_Correlation_Matrix.png")

    # Remove features with high correlation
    logging.info(f"Removing highly correlated features: {config.HIGHLY_CORRELATED_FEATURES}")
    features = features.drop(config.HIGHLY_CORRELATED_FEATURES, axis=1)
    
    ### Feature selection
    # PCA
    pca, x_pca = Statistics.get_pca_transformed_data(features, n_components=2)
    Visualization.plot_pca_components(x_pca, target, output_folder=visualization_folder, output_filename=f"{args.input_spreadsheet.split('.')[0]}_PCA_components.png")

    pca_loadings = FeatureImportance.get_pca_loadings_first_component(pca, features.columns)
    Visualization.plot_features_importance(pca_loadings, output_folder=visualization_folder, output_filename=f"{args.input_spreadsheet.split('.')[0]}_PCA_loading_scores.png", title="PCA loading scores")

    # Random Forest
    rf_importances = FeatureImportance.get_random_forest_features_importance(features, target)
    Visualization.plot_features_importance(rf_importances, output_folder=visualization_folder, output_filename=f"{args.input_spreadsheet.split('.')[0]}_Random_Forest.png", title="Random Forest")

    # XGBoost
    xgb_importances = FeatureImportance.get_xgboost_features_importance(features, target)
    Visualization.plot_features_importance(xgb_importances, output_folder=visualization_folder, output_filename=f"{args.input_spreadsheet.split('.')[0]}_XGBoost.png", title="XGBoost")

    # Logistic Regression
    lr_importances = FeatureImportance.get_logistic_regression_features_importance(features, target)
    Visualization.plot_features_importance(lr_importances, output_folder=visualization_folder, output_filename=f"{args.input_spreadsheet.split('.')[0]}_Logistic_Regression.png", title="Logistic Regression")

    # ANOVA
    anova_importances = FeatureImportance.get_anova_f_scores(features, target)
    Visualization.plot_features_importance(anova_importances, output_folder=visualization_folder, output_filename=f"{args.input_spreadsheet.split('.')[0]}_ANOVA_F-scores.png", title="ANOVA F scores")

    # Chi-squared
    chi_squared_importances = FeatureImportance.get_chi_squared_stats(features, target)
    Visualization.plot_features_importance(chi_squared_importances, output_folder=visualization_folder, output_filename=f"{args.input_spreadsheet.split('.')[0]}_Chi-squared_Statistics.png", title="Chi-squared statistics")

    # Append the method to columns for further merging
    pca_loadings.rename({'Importance': 'PCA loadings Importance'}, axis=1, inplace=True)
    rf_importances.rename({'Importance': 'Random Forest Importance'}, axis=1, inplace=True)
    xgb_importances.rename({'Importance': 'XGBoost Importance'}, axis=1, inplace=True)
    lr_importances.rename({'Importance': 'Logistic Regression Importance'}, axis=1, inplace=True)
    anova_importances.rename({'Importance': 'ANOVA F scores Importance'}, axis=1, inplace=True)
    chi_squared_importances.rename({'Importance': 'Chi-squared statistics Importance'}, axis=1, inplace=True)

    # For each importance method, apply min-max scaling to obtain comparable values
    pca_loadings["PCA loadings Scaled Importance"] = Statistics.min_max_scaler(pca_loadings[['PCA loadings Importance']])
    rf_importances["Random Forest Scaled Importance"] = Statistics.min_max_scaler(rf_importances[['Random Forest Importance']])
    xgb_importances["XGBoost Scaled Importance"] = Statistics.min_max_scaler(xgb_importances[['XGBoost Importance']])
    lr_importances["Logistic Regression Scaled Importance"] = Statistics.min_max_scaler(lr_importances[['Logistic Regression Importance']])
    anova_importances["ANOVA F scores Scaled Importance"] = Statistics.min_max_scaler(anova_importances[['ANOVA F scores Importance']])
    chi_squared_importances["Chi-squared statistics Scaled Importance"] = Statistics.min_max_scaler(chi_squared_importances[['Chi-squared statistics Importance']])

    # Save all importances to CSV
    IO.save_dataframe(pca_loadings, spreadsheets_folder, f"{args.input_spreadsheet.split('.')[0]}_pca_loadings.csv")
    IO.save_dataframe(rf_importances, spreadsheets_folder, f"{args.input_spreadsheet.split('.')[0]}_rf_importances.csv")
    IO.save_dataframe(xgb_importances, spreadsheets_folder, f"{args.input_spreadsheet.split('.')[0]}_xgb_importances.csv")
    IO.save_dataframe(lr_importances, spreadsheets_folder, f"{args.input_spreadsheet.split('.')[0]}_lr_importances.csv")
    IO.save_dataframe(anova_importances, spreadsheets_folder, f"{args.input_spreadsheet.split('.')[0]}_anova_importances.csv")
    IO.save_dataframe(chi_squared_importances, spreadsheets_folder, f"{args.input_spreadsheet.split('.')[0]}_chi_squared_importances.csv")

    # Merge all importances into a single dataframe
    all_importances = reduce(lambda  left,right: pd.merge(left,right,on=['Feature'], how='outer'), [pca_loadings, rf_importances, xgb_importances, lr_importances, anova_importances, chi_squared_importances])
    IO.save_dataframe(all_importances, spreadsheets_folder, f"{args.input_spreadsheet.split('.')[0]}_all_importances.csv")