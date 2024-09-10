import os
import zipfile
import pandas as pd

def get_kaggle_data(folder_path, dataset_name):
    """
    Downloads and extracts a Kaggle dataset to a specified folder.

    Parameters:
    folder_path (str): Path to the folder where the dataset should be downloaded.
    dataset_name (str): Name of the dataset on Kaggle.
    """

    os.makedirs(folder_path, exist_ok=True)

    # Download the dataset from Kaggle
    result = os.system(f"kaggle competitions download -c {dataset_name}")
    if result != 0:
        raise Exception("Error downloading dataset. Please check if Kaggle API is configured.")
    
    zip_file = f'{dataset_name}.zip'
    if not os.path.exists(zip_file):
        raise Exception(f"Failed to find the zip file {zip_file}. Check if the dataset name is correct.")
    
    # Extract the downloaded zip file
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(folder_path)

    # Remove the zip file after extraction
    os.remove(zip_file)

def load_data(folder_path, file_name):
    """
    Loads a CSV file from a specified folder.

    Parameters:
    folder_path (str): Path to the folder containing the CSV file.
    file_name (str): Name of the CSV file.

    Returns:
    pd.DataFrame: DataFrame containing the data from the CSV file.
    """
    file_path = os.path.join(folder_path, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_name} not found in {folder_path}")

    return pd.read_csv(file_path)

def kaggle_download_and_load(folder_path, dataset_name):
    """
    Downloads and loads a Kaggle dataset into a DataFrame.

    Parameters:
    folder_path (str): Path to the folder where the dataset should be extracted.
    dataset_name (str): Name of the dataset on Kaggle.

    Returns:
    (pd.DataFrame, pd.DataFrame): Train and Test DataFrames.
    """
    get_kaggle_data(folder_path, dataset_name)
    train_data = load_data(folder_path, "train.csv")
    test_data = load_data(folder_path, "test.csv")
    return train_data, test_data