import pandas as pd
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, LeavePOut, StratifiedKFold

def hold_out_split(csv_file_path, params, random_state=42):
    """
    Performs a hold-out cross-validation split on a dataset.

    Parameters:
    - csv_file_path (str): Path to the CSV file containing the dataset.
    - test_size (float): Proportion of the dataset to include in the test split (default is 0.2).
    - random_state (int): Controls the shuffling applied to the data before splitting (default is 42).

    Returns:
    - X_train (DataFrame): Training set features.
    - X_test (DataFrame): Test set features.
    - y_train (Series): Training set labels.
    - y_test (Series): Test set labels.
    """
    test_size=params['test_size']
    # Load dataset
    data = pd.read_csv(csv_file_path)
    
    # Assuming the last column is the target variable
    X = data.iloc[:, :-1]  # All columns except the last as features
    y = data.iloc[:, -1]   # The last column as the target variable
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test


def get_k_folds_splits(csv_file_path, params, random_state=42):
    """
    Performs K-Folds Cross-Validation split on a dataset and returns the train-test splits.

    Parameters:
    - csv_file_path (str): Path to the CSV file containing the dataset.
    - n_splits (int): Number of folds for K-Folds Cross-Validation (default is 5).
    - random_state (int): Controls shuffling before cross-validation split (default is 42).

    Returns:
    - folds (list): A list of tuples where each tuple contains:
        - X_train (DataFrame): Training set features for the fold.
        - X_test (DataFrame): Test set features for the fold.
        - y_train (Series): Training set labels for the fold.
        - y_test (Series): Test set labels for the fold.
    """
    n_splits=params['n_splits']
    # Load dataset    
    data = pd.read_csv(csv_file_path)
    # Assuming the last column is the target variable
    X = data.iloc[:, :-1]  # All columns except the last as features
    y = data.iloc[:, -1]   # The last column as the target variable
    
    # Initialize KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # List to store each fold's train-test split
    folds = []
    
    # Perform K-Folds Cross-Validation splitting
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Append the split data for each fold
        folds.append((X_train, X_test, y_train, y_test))
    
    return folds


def get_leave_one_out_splits(csv_file_path, params):
    """
    Performs Leave-One-Out Cross-Validation (LOOCV) split on a dataset and returns the train-test splits.

    Parameters:
    - csv_file_path (str): Path to the CSV file containing the dataset.

    Returns:
    - splits (list): A list of tuples where each tuple contains:
        - X_train (DataFrame): Training set features for the fold.
        - X_test (DataFrame): Test set feature (single sample) for the fold.
        - y_train (Series): Training set labels for the fold.
        - y_test (Series): Test set label (single sample) for the fold.
    """
    # Load dataset
    data = pd.read_csv(csv_file_path)
    
    # Assuming the last column is the target variable
    X = data.iloc[:, :-1]  # All columns except the last as features
    y = data.iloc[:, -1]   # The last column as the target variable
    
    # Initialize LeaveOneOut
    loo = LeaveOneOut()
    
    # List to store each fold's train-test split
    splits = []
    
    # Perform LOOCV splitting
    for train_index, test_index in loo.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Append the split data for each fold
        splits.append((X_train, X_test, y_train, y_test))
    
    return splits


def get_leave_p_out_splits(csv_file_path, params):
    """
    Performs Leave-p-Out Cross-Validation (LpOCV) split on a dataset and returns the train-test splits.

    Parameters:
    - csv_file_path (str): Path to the CSV file containing the dataset.
    - p (int): Number of samples to leave out for each test set.

    Returns:
    - splits (list): A list of tuples where each tuple contains:
        - X_train (DataFrame): Training set features for the fold.
        - X_test (DataFrame): Test set features (p samples) for the fold.
        - y_train (Series): Training set labels for the fold.
        - y_test (Series): Test set labels (p samples) for the fold.
    """
    p=params['p']
    # Load dataset
    data = pd.read_csv(csv_file_path)
    
    # Assuming the last column is the target variable
    X = data.iloc[:, :-1]  # All columns except the last as features
    y = data.iloc[:, -1]   # The last column as the target variable
    
    # Initialize LeavePOut with specified p
    lpo = LeavePOut(p)
    
    # List to store each fold's train-test split
    splits = []
    
    # Perform Leave-p-Out splitting
    for train_index, test_index in lpo.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Append the split data for each fold
        splits.append((X_train, X_test, y_train, y_test))
    
    return splits


def stratified_kfold_cv(csv_file, params, random_state=None):
    n_splits=params['n_splits']
    shuffle=params['shuffle']
    # Load dataset
    df = pd.read_csv(csv_file)
    
    # Assume target variable is in the last column
    X = df.iloc[:, :-1].values  # Features
    y = df.iloc[:, -1].values   # Target variable
    
    # Initialize Stratified K-Folds
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    
    # Prepare list to store the CV splits
    cv_splits = []
    
    # Perform Stratified K-Folds split
    for train_index, test_index in skf.split(X, y):
        # Split the dataset
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Store the split data
        cv_splits.append({
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        })
    
    return cv_splits


def repeated_kfold_cv(csv_file, params, random_state=None):
    n_splits=params['n_splits']
    shuffle=params['shuffle']
    n_repeats=params['n_repeats']
    # Load dataset
    df = pd.read_csv(csv_file)
    
    # Assume target variable is in the last column
    X = df.iloc[:, :-1].values  # Features
    y = df.iloc[:, -1].values   # Target variable
    
    # Initialize K-Folds (to be repeated)
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    
    # Prepare list to store the CV splits
    cv_splits = []
    
    # Perform repeated K-Folds
    for repeat in range(n_repeats):
        for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
            # Split the dataset
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # Store the split data
            cv_splits.append({
                'repeat': repeat + 1,
                'fold': fold + 1,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            })
    
    return cv_splits


def nested_kfold_split(csv_file, params, random_state=None):
    n_splits=params['n_splits']
    shuffle=params['shuffle']
    n_repeats=params['n_repeats']
    # Load dataset
    df = pd.read_csv(csv_file)
    
    # Assume target variable is in the last column
    X = df.iloc[:, :-1].values  # Features
    y = df.iloc[:, -1].values   # Target variable
    
    # Initialize outer K-Fold
    outer_kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    
    # Prepare list to store splits
    splits = []
    
    # Outer loop for evaluation
    for repeat in range(n_repeats):
        for outer_train_index, outer_test_index in outer_kf.split(X, y):
            # Split dataset for the outer fold
            X_train_outer, X_test_outer = X[outer_train_index], X[outer_test_index]
            y_train_outer, y_test_outer = y[outer_train_index], y[outer_test_index]
            
            # Initialize inner K-Fold for cross-validation within the outer training set
            inner_kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
            
            # Store the splits for the inner K-Fold (for each outer training set)
            inner_splits = []
            for inner_train_index, inner_test_index in inner_kf.split(X_train_outer, y_train_outer):
                # Inner fold splits
                X_train_inner, X_test_inner = X_train_outer[inner_train_index], X_train_outer[inner_test_index]
                y_train_inner, y_test_inner = y_train_outer[inner_train_index], y_train_outer[inner_test_index]
                inner_splits.append({
                    'X_train': X_train_inner, 'X_test': X_test_inner,
                    'y_train': y_train_inner, 'y_test': y_test_inner
                })
            
            # Store the outer fold splits along with the inner fold splits
            splits.append({
                'repeat': repeat + 1,
                'outer_train': (X_train_outer, y_train_outer),
                'outer_test': (X_test_outer, y_test_outer),
                'inner_splits': inner_splits
            })
    
    return splits
