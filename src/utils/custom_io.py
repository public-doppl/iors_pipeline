from pathlib import Path
import pandas as pd
import os

def validate_inputs(input_folder: str, input_spreadsheet: str):
    """Validate the input folder and spreadsheet."""
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder '{input_folder}' does not exist.")
    if not os.path.isfile(Path(input_folder) / input_spreadsheet):
        raise FileNotFoundError(f"Input file '{input_spreadsheet}' does not exist in '{input_folder}'.")
    if not input_spreadsheet.endswith(('.csv', '.xlsx')):
        raise ValueError("Input file must be a .csv or .xlsx file.")

def load_dataset(input_folder: str, input_spreadsheet: str):
    """Load the dataset from the specified folder and spreadsheet."""
    validate_inputs(input_folder, input_spreadsheet)
    if input_spreadsheet.endswith('.xlsx'):
        return pd.read_excel(Path(input_folder) / input_spreadsheet, engine='openpyxl')
    elif input_spreadsheet.endswith('.csv'):
        return pd.read_csv(Path(input_folder) / input_spreadsheet)

def create_output_folders(output_folder: str):
    """Create output folders for visualizations and spreadsheets."""
    output_folder = Path(output_folder)
    create_folder(output_folder)
    visualization_folder = output_folder / "visualizations"
    create_folder(visualization_folder)
    spreadsheets_folder = output_folder / "spreadsheets"
    create_folder(spreadsheets_folder)
    return visualization_folder, spreadsheets_folder

def create_folder(folder_path: Path):
    """Create a folder if it does not exist."""
    folder_path.mkdir(parents=True, exist_ok=True)

def save_dataframe(df: pd.DataFrame, folder: Path, filename: str):
    """Save the DataFrame to a CSV file in the specified folder."""
    df.to_csv(folder / filename, index=False)