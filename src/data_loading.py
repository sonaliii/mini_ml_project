import pandas as pd


def load_data(csv_path):
    """
    Load input CSV into a pandas DataFrame
    :param csv_path: str absolute path to csv file
    :return: pandas DataFrame with input data
    """
    df = pd.read_csv(csv_path)
    return df
