import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import utils.config as config

def set_prism_style():
    """
    Set the style for the plots to match the Prism style.
    """
    sns.set_style("whitegrid")
    sns.set_context("talk", font_scale=1.2)
    sns.set_style("whitegrid", config.MATPLOTLIB_PRISM_STYLE)

def plot_correlation_matrix(df, output_folder, output_filename):
    """
    Plot the correlation matrix of the DataFrame and save it to a file.
    Args:
        df (pd.DataFrame): The DataFrame to plot.
        output_folder (str): The folder to save the plot.
        output_filename (str): The filename for the saved plot.
    Returns:
        pd.DataFrame: The correlation matrix.
    """
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    set_prism_style()
    corr = df.corr()

    params = {'axes.titlesize':'15',
        'xtick.labelsize':'10',
        'ytick.labelsize':'10'}

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(
        figsize=(18, 18)
    )

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(270, 310, s=85, l=25, as_cmap=True)

    plt.title('Correlation matrix')

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
    ax.grid(False) 
    plt.savefig(Path(output_folder) / output_filename, dpi=300, bbox_inches='tight')
    return corr

def plot_pca_components(x_pca, target, output_folder, output_filename):
    """
    Plot the PCA components and save the plot to a file.
    Args:
        x_pca (np.ndarray): The PCA transformed data.
        target (pd.Series): The target variable.
        output_folder (str): The folder to save the plot.
        output_filename (str): The filename for the saved plot.
    """
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    set_prism_style()

    colormap = {'Control': 'black', 'CytoMix':'purple'}
    colors = target.map(colormap)

    plt.figure(figsize=(8,6))
    plt.scatter(x_pca[:,0],x_pca[:,1],c=colors)
    plt.xlabel('First principal component')
    plt.ylabel('Second Principal Component')

    # Add legend
    for group, color in colormap.items():
        plt.scatter([], [], c=color, label=group)  # Add invisible points for legend
    plt.legend(title="Target", loc="best")
    plt.savefig(Path(output_folder) / output_filename, dpi=300, bbox_inches='tight')
    return

def plot_features_importance(importances, output_folder, output_filename, title=None, log=False):
    """
    Plot the feature importances and save the plot to a file.
    Args:
        importances (pd.DataFrame): The feature importances DataFrame.
        output_folder (str): The folder to save the plot.
        output_filename (str): The filename for the saved plot.
        title (str): The title of the plot.
        log (bool): Whether to use a logarithmic scale for the y-axis.
    """
    importances.sort_values(by='Importance', ascending=False, inplace=True)
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)

    set_prism_style()

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.barplot(data=importances, x='Feature', y='Importance', ax=ax, palette="Purples", hue='Feature')
    if not title:
        ax.set_title("Features importance")
    else:
        ax.set_title(f"Features importance - {title}")
    if log:
        ax.set_yscale('log')
    ax.set_ylabel("Importance")
    ax.set_xlabel("Feature")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(Path(output_folder) / output_filename, dpi=300, bbox_inches='tight')
    return importances

def plot_recovery_score_boxplot(dataset, output_folder, output_prefix=""):
    """
    Plot the recovery score boxplot and save it to a file.
    Args:
        dataset (pd.DataFrame): The dataset containing the recovery scores.
        output_folder (str): The folder to save the plot.
        output_prefix (str): The prefix for the output filename.
    """
    set_prism_style()
    dataset_plot = dataset.copy()
    sns.boxplot(x=dataset_plot['Condition'], y=dataset_plot['Recovery_score'], showfliers=True, palette='Purples', legend=False, hue=dataset_plot['Condition'])
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.title('Recovery Score by Condition', fontsize=16)
    plt.ylabel('Recovery Score', fontsize=14)
    plt.xlabel('Condition', fontsize=14)
    plt.tight_layout()
    plt.savefig(Path(output_folder) / f"{output_prefix}recovery_score.png", dpi=300, bbox_inches='tight')