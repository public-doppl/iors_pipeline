import argparse
import os
import pandas as pd
from pathlib import Path

import utils.config as config
import utils.custom_io as IO
import utils.preprocessing_recovery_score as Preprocessing
import utils.visualization as Visualization
import utils.custom_statistics as Statistics

def compute_A_factor(x, area_columns):
    """
    Computes the A factor for a given organoid and area columns.
    The A factor is the area under the curve (AUC) of the organoid's area over time.
    Args:
        x (pd.Series): A row of the pivoted DataFrame representing an organoid's area values at different time points.
        area_columns (list): List of columns representing the area over time.
    Returns:
        float: The A factor for the organoid.
    """
    return compute_AUC(x, area_columns)

def compute_B_factor(x, intensity_columns):
    """
    Computes the B factor for a given organoid and intensity columns.
    The B factor is the area under the curve (AUC) of the organoid's intensity over time.
    Args:
        x (pd.Series): A row of the pivoted DataFrame representing an organoid's intensity values at different time points.
        intensity_columns (list): List of columns representing the intensity over time.
    Returns:
        float: The B factor for the organoid.
    """
    return compute_AUC(x, intensity_columns)

def compute_AUC(organoid, timepoints):
    """
    Computes the area under the curve (AUC) for a given organoid over time.
    In this implementation of the AUC computation, area of curve below intercept negatively contributes to the AUC.
    Args:
        organoid (pd.Series): A row of the pivoted DataFrame representing an organoid's values at different time points.
        timepoints (list): List of columns representing the time points.
    Returns:
        float: The AUC for the organoid.
    """
    organoid = organoid[timepoints]
    days = [int(time[1].replace('T','')) for time in timepoints]
    y_intercept = 1
    corresponding_x = [days[i]-days[0] for i in range(len(days))]
    timepoints_steps = [corresponding_x[i+1] - corresponding_x[i] for i in range(len(days) - 1)]

    total_AUC = 0

    for i in range(len(timepoints_steps)):
        if (organoid[timepoints[i]] >= y_intercept) and (organoid[timepoints[i+1]] >= y_intercept):
            total_AUC += (organoid[timepoints[i+1]] + organoid[timepoints[i]] - 2*y_intercept)*timepoints_steps[i]/2
        elif (organoid[timepoints[i]] < y_intercept) and (organoid[timepoints[i+1]] < y_intercept):
            negative_area = (2*y_intercept - organoid[timepoints[i+1]] - organoid[timepoints[i]])*timepoints_steps[i]/2
            total_AUC-= negative_area
        elif (organoid[timepoints[i]] < y_intercept):
            x_intercept = find_interpolation_intercept(organoid[timepoints[i+1]], organoid[timepoints[i]], corresponding_x[i+1], corresponding_x[i], y_intercept)
            positive_area = (organoid[timepoints[i+1]] -y_intercept)*(corresponding_x[i+1] - x_intercept)/2
            total_AUC += positive_area
            negative_area = (organoid[timepoints[i]] - y_intercept)*(x_intercept - corresponding_x[i])/2
            total_AUC -= negative_area
        elif (organoid[timepoints[i+1]] < y_intercept):
            x_intercept = find_interpolation_intercept(organoid[timepoints[i+1]], organoid[timepoints[i]], corresponding_x[i+1], corresponding_x[i], y_intercept)
            positive_area = (organoid[timepoints[i]] - y_intercept)*(x_intercept - corresponding_x[i])/2
            total_AUC += positive_area
            negative_area = (organoid[timepoints[i+1]] -y_intercept)*(corresponding_x[i+1] - x_intercept)/2
            total_AUC -= negative_area

    return total_AUC

def find_interpolation_intercept(y1, y0, x1, x0, y_intercept):
    """
    Finds the x-intercept of the line connecting two points (x0, y0) and (x1, y1) at a given y-intercept.
    Args:
        y1 (float): The y-coordinate of the second point.
        y0 (float): The y-coordinate of the first point.
        x1 (float): The x-coordinate of the second point.
        x0 (float): The x-coordinate of the first point.
        y_intercept (float): The y-intercept to find the corresponding x value for.
    Returns:
        float: The x-coordinate of the intersection with the y-intercept.
    """
    a = (y1-y0) / (x1-x0)
    b = y0 - a*x0
    x_intercept = (y_intercept - b) / a
    return x_intercept

def compute_recovery_score(pivot_dataset):
    """
    Computes the recovery score for each organoid in the pivoted dataset.
    The recovery score is computed as the sum of the A factor and B factor for each organoid.
    Args:
        pivot_dataset (pd.DataFrame): The pivoted DataFrame containing organoid data.
    Returns:
        pd.DataFrame: The DataFrame with the recovery score for each organoid.
    """
    dataset_with_recovery_score = pivot_dataset.copy()
    area_columns = [col for col in dataset_with_recovery_score.columns if col[0]=='Area (um2)']
    dataset_with_recovery_score['A_factor'] = dataset_with_recovery_score.apply(lambda x: compute_A_factor(x, area_columns), axis=1)
    intensity_columns = [col for col in dataset_with_recovery_score.columns if col[0]=='Normalized mean intensity']
    dataset_with_recovery_score['B_factor'] = dataset_with_recovery_score.apply(lambda x: compute_B_factor(x, intensity_columns), axis=1)
    outliers = Statistics.is_an_outlier(dataset_with_recovery_score, ['A_factor', 'B_factor'])
    dataset_with_recovery_score = dataset_with_recovery_score[~ outliers]
    dataset_with_recovery_score['Recovery_score_unscaled'] = dataset_with_recovery_score['A_factor'] + dataset_with_recovery_score['B_factor']
    dataset_with_recovery_score['Recovery_score'] = Statistics.min_max_scaler(dataset_with_recovery_score['Recovery_score_unscaled'].values.reshape(-1, 1))
    return dataset_with_recovery_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Feature selection pipeline",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_folder", help="input folder", required=True)
    parser.add_argument("--input_spreadsheet", help="input spreadsheet filename", required=True)
    parser.add_argument("--output_folder", help="output folder where stacks should be stored", required=True)
    args = parser.parse_args()

    ### Loading the dataset
    raw_dataset = IO.load_dataset(args.input_folder, args.input_spreadsheet)

    ### Creating output folders
    visualization_folder, spreadsheets_folder = IO.create_output_folders(args.output_folder)

    ### Preprocessing the dataset
    pivot_dataset = Preprocessing.preprocess_dataset(raw_dataset)

    ### Computing IORS
    IORS_dataset = compute_recovery_score(pivot_dataset).reset_index()
    IO.save_dataframe(IORS_dataset, spreadsheets_folder, f"{args.input_spreadsheet.split('.')[0]}_IORS.csv")

    ### Plotting results for main conditions of interest
    IORS_dataset_selected_conditions = IORS_dataset[IORS_dataset['Condition'].isin(config.IORS_TEST_CONDITIONS)]

    Visualization.plot_recovery_score_boxplot(IORS_dataset_selected_conditions, visualization_folder, output_prefix=f"{args.input_spreadsheet.split('.')[0]}_")

