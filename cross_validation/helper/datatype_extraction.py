
import pandas as pd

def get_feature_data_types(file_path):
    """
    Function to read a data file in CSV, XLSX, or XML format and return the data types of all features.
    
    Parameters:
        file_path (str): Path to the input file (CSV, XLSX, or XML).

    Returns:
        dict: A dictionary where keys are feature names and values are their data types.
    """
    # Determine file format based on extension
    file_extension = file_path.split('.')[-1].lower()

    # Read the file into a pandas DataFrame
    if file_extension == 'csv':
        df = pd.read_csv(file_path)
    elif file_extension == 'xlsx':
        df = pd.read_excel(file_path)
    elif file_extension == 'xml':
        df = pd.read_xml(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a file in CSV, XLSX, or XML format.")

    # Get data types of each column
    data_types = df.dtypes

    # Convert to dictionary with feature names as keys and data types as values
    feature_types = {col: str(dtype) for col, dtype in data_types.items()}

    return feature_types

def get_target_data_types(file_path, target_column):
    """
    Function to read a data file in CSV, XLSX, or XML format and return the data type of the target column.
    
    Parameters:
        file_path (str): Path to the input file (CSV, XLSX, or XML).
        target_column (str): The name of the target column.

    Returns:
        str: The data type of the target column.
    """
    # Determine file format based on extension
    file_extension = file_path.split('.')[-1].lower()

    # Read the file into a pandas DataFrame
    if file_extension == 'csv':
        df = pd.read_csv(file_path)
    elif file_extension == 'xlsx':
        df = pd.read_excel(file_path)
    elif file_extension == 'xml':
        df = pd.read_xml(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a file in CSV, XLSX, or XML format.")

    # Check if the target column exists in the DataFrame
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the data file.")

    # Get the data type of the target column
    target_type = df[target_column].dtype

    return str(target_type)
