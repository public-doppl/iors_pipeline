# IORS pipeline

This repository contains a pipeline for feature selection and IORS (Intestinal Organoid Recovery Score) computation. The pipeline is containerized using Docker for easy deployment and reproducibility.

## Prerequisites

1. Install Docker on your system.

2. Ensure that your parameters.json file is properly configured with the required input and output paths.

## Configuring parameters.json
The parameters.json file (located at the root directory of the repository) should be structured as an array of objects, where each object specifies the input folder, input spreadsheet (csv or xlsx), and output folder. For example:

```json
[
    {
        "input_folder": "./data/input",
        "input_spreadsheet": "dataset1.xlsx",
        "output_folder": "./data/output"
    },
    {
        "input_folder": "./data/input",
        "input_spreadsheet": "dataset2.xlsx",
        "output_folder": "./data/output"
    }
]
```
Make sure to provide your input dataset in the corresponding folder.

## Building the Docker image
To build the Docker image, navigate to the root directory of the repository (where the Dockerfile is located) and run the following command:
```bash
docker build -t iors_pipeline .
```
* *iors_pipeline* is the name of the Docker image. You can replace it with any name you prefer.

## Running the Docker container
To run the pipeline, , navigate to the root directory of the repository and use the following command.
```bash
docker run --rm -v $(pwd):/app iors_pipeline
```
* *iors_pipeline* is the name of the Docker image built at the previous step.

## Accessing the Output

After the container finishes running, the output files will be available in the data/output directory on your host machine.