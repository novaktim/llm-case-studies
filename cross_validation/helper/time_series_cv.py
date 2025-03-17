import pandas as pd

def rolling_window_partition_df(data, train_size, test_size):
    """
    Partition a time series dataframe into rolling train-test splits.

    Parameters:
    - data (pd.DataFrame): Time series data to be partitioned.
    - train_size (int): Number of observations in each training window.
    - test_size (int): Number of observations in each test window.

    Returns:
    - partitions (list of tuples): List of (train_df, test_df) tuples for each rolling split.
    """
    partitions = []
    n_splits = len(data) - train_size - test_size + 1
    
    for i in range(n_splits):
        # Define the train and test sets for the current split
        train_df = data.iloc[i: i + train_size].reset_index(drop=True)
        test_df = data.iloc[i + train_size: i + train_size + test_size].reset_index(drop=True)
        
        # Append the split as a tuple
        partitions.append((train_df, test_df))
    
    return partitions


