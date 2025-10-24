# Setting Up the Conda Environment

Follow these steps to create and activate a Conda environment using the provided `environment.yml` file.

## 1. Locate the `environment.yml` File
Ensure you have an `environment.yml` file in your project directory. This file specifies the dependencies and configurations for the Conda environment.

## 2. Create the Environment
Run the following command to create a Conda environment from the `.yml` file:

```bash
conda env create -f environment.yml
