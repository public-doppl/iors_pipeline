#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# PARAMETERS_FILE="$1"
PARAMETERS_FILE="parameters.json"

# Check if the JSON parameters file exists
if [[ ! -f "$PARAMETERS_FILE" ]]; then
    echo "Error: Parameters file '$PARAMETERS_FILE' not found."
    exit 1
fi

# Parse the JSON file and extract input and output paths
entries=$(jq -c '.[]' "$PARAMETERS_FILE")

# Iterate over each entry in the JSON file
for entry in $entries; do
    input_folder=$(echo "$entry" | jq -r '.input_folder')
    input_spreadsheet=$(echo "$entry" | jq -r '.input_spreadsheet')
    output_folder=$(echo "$entry" | jq -r '.output_folder')
    # feature_selection_cytokine_days=$(echo "$entry" | jq -r '.feature_selection_cytokine_days')
    # feature_selection_outlier_conditions=$(echo "$entry" | jq -r '.feature_selection_outlier_conditions')

    # Skip if missing arguments
    if [[ "$input_folder" == "null" || "$input_spreadsheet" == "null" || "$output_folder" == "null" ]]; then
        echo "Skipping entry with missing input_folder or input_spreadsheet or output_folder."
        continue
    fi

    # Run the feature selection script
    echo "Running feature selection..."
    python src/feature_selection.py --input_folder "$input_folder" --input_spreadsheet "$input_spreadsheet" --output_folder "$output_folder"

    # Run the IORS script
    echo "Running IORS computation..."
    python src/IORS.py --input_folder "$input_folder" --input_spreadsheet "$input_spreadsheet" --output_folder "$output_folder"
done

echo "All analyses completed successfully."