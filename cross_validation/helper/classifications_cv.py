import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, LeavePOut, StratifiedKFold
from sklearn.metrics import mean_absolute_error, accuracy_score
from classes.CV_Response import CrossValidationResponse
from classes.ProblemType import ProblemType

def hold_out_split_and_train(csv_file_path, params, model, target_variable_type=ProblemType.REGRESSION.value, random_state=42):
    """
    Performs a hold-out cross-validation split on a dataset and trains a model.

    Parameters:
    - csv_file_path (str): Path to the CSV file containing the dataset.
    - test_size (float): Proportion of the dataset to include in the test split.
    - model: The machine learning model to be trained.
    - random_state (int): Controls the shuffling applied to the data before splitting (default is 42).

    Returns:
    - model: The trained machine learning model.
    - X_train (DataFrame): Training set features.
    - X_test (DataFrame): Test set features.
    - y_train (Series): Training set labels.
    - y_test (Series): Test set labels.
    """
    test_size = params['test_size']
    # Load dataset
    data = pd.read_csv(csv_file_path)
    
    # Assuming the last column is the target variable
    X = data.iloc[:, 1:-1]  # All columns except the first and the last as features
    y = data.iloc[:, -1]   # The last column as the target variable
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Train the model
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    if target_variable_type == ProblemType.CLASSIFICATION.value:
        accuracy = accuracy_score(y_test, y_pred)
    elif target_variable_type == ProblemType.REGRESSION.value:
        accuracy = mean_absolute_error(y_test, y_pred)
    else:
        raise ValueError("Invalid target_variable_type. Expected 'classification' or 'regression'.")
    
    return CrossValidationResponse('Hold-Out', params, accuracy, model)


def get_k_folds_splits_and_train(csv_file_path, params, model,target_variable_type=ProblemType.REGRESSION.value, random_state=42):
    """
    Performs K-Folds Cross-Validation split on a dataset, trains a model on each fold, and returns the trained models and their accuracies.

    Parameters:
    - csv_file_path (str): Path to the CSV file containing the dataset.
    - n_splits (int): Number of folds for K-Folds Cross-Validation (default is 5).
    - model: The machine learning model to be trained.
    - random_state (int): Controls shuffling before cross-validation split (default is 42).

    Returns:
    - results (list): A list of dictionaries where each dictionary contains:
        - 'fold': The fold number.
        - 'model': The trained machine learning model for the fold.
        - 'accuracy': The accuracy of the model on the test set for the fold.
    """
    n_splits = params['n_splits']
    # Load dataset    
    data = pd.read_csv(csv_file_path)
    # Assuming the last column is the target variable
    X = data.iloc[:, :-1]  # All columns except the last as features
    y = data.iloc[:, -1]   # The last column as the target variable
    
    # Initialize KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # List to store results for each fold
    results = []
    fold_accuracy = []
    
    # Perform K-Folds Cross-Validation splitting and training
    for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Train the model
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        if target_variable_type == ProblemType.CLASSIFICATION.value:
            fold_accuracy.append(accuracy_score(y_test, y_pred))
        elif target_variable_type == ProblemType.REGRESSION.value:
            fold_accuracy.append(mean_absolute_error(y_test, y_pred))
        else:
            raise ValueError("Invalid target_variable_type. Expected 'classification' or 'regression'.")
    
    return CrossValidationResponse('K-Folds', params, np.mean(fold_accuracy), model)


def get_leave_one_out_splits_and_train(csv_file_path, params, model, target_variable_type=ProblemType.REGRESSION.value):
    """
    Performs Leave-One-Out Cross-Validation (LOOCV) split on a dataset, trains a model on each fold, and returns the trained models and their accuracies.

    Parameters:
    - csv_file_path (str): Path to the CSV file containing the dataset.
    - model: The machine learning model to be trained.

    Returns:
    - results (list): A list of dictionaries where each dictionary contains:
        - 'fold': The fold number.
        - 'model': The trained machine learning model for the fold.
        - 'accuracy': The accuracy of the model on the test set for the fold.
    """
    # Load dataset
    data = pd.read_csv(csv_file_path)
    
    # Assuming the last column is the target variable
    X = data.iloc[:, :-1]  # All columns except the last as features
    y = data.iloc[:, -1]   # The last column as the target variable
    
    # Initialize LeaveOneOut
    loo = LeaveOneOut()
    
    # List to store results for each fold
    results = []
    fold_accuracy = []
    
    # Perform LOOCV splitting and training
    for fold, (train_index, test_index) in enumerate(loo.split(X), 1):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Train the model
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        if target_variable_type == ProblemType.CLASSIFICATION.value:
            fold_accuracy.append(accuracy_score(y_test, y_pred))
        elif target_variable_type == ProblemType.REGRESSION.value:
            fold_accuracy.append(mean_absolute_error(y_test, y_pred))
        else:
            raise ValueError("Invalid target_variable_type. Expected 'classification' or 'regression'.")
    
    return CrossValidationResponse('Leave-One-Out', params, np.mean(fold_accuracy), model)


def get_leave_p_out_splits_and_train(csv_file_path, params, model, target_variable_type=ProblemType.REGRESSION.value):
    """
    Performs Leave-p-Out Cross-Validation (LpOCV) split on a dataset, trains a model on each fold, and returns the trained models and their accuracies.

    Parameters:
    - csv_file_path (str): Path to the CSV file containing the dataset.
    - p (int): Number of samples to leave out for each test set.
    - model: The machine learning model to be trained.

    Returns:
    - results (list): A list of dictionaries where each dictionary contains:
        - 'fold': The fold number.
        - 'model': The trained machine learning model for the fold.
        - 'accuracy': The accuracy of the model on the test set for the fold.
    """
    p = params['p']
    # Load dataset
    data = pd.read_csv(csv_file_path)
    
    # Assuming the last column is the target variable
    X = data.iloc[:, :-1]  # All columns except the last as features
    y = data.iloc[:, -1]   # The last column as the target variable
    
    # Initialize LeavePOut with specified p
    lpo = LeavePOut(p)
    
    # List to store results for each fold
    results = []
    fold_accuracy = []
    
    # Perform Leave-p-Out splitting and training
    for fold, (train_index, test_index) in enumerate(lpo.split(X), 1):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Train the model
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        if target_variable_type == ProblemType.CLASSIFICATION.value:
            fold_accuracy.append(accuracy_score(y_test, y_pred))
        elif target_variable_type == ProblemType.REGRESSION.value:
            fold_accuracy.append(mean_absolute_error(y_test, y_pred))
        else:
            raise ValueError("Invalid target_variable_type. Expected 'classification' or 'regression'.")
    
    return CrossValidationResponse('Leave-p-Out', params, np.mean(fold_accuracy), model)


def stratified_kfold_cv(csv_file, params, model,target_variable_type=ProblemType.REGRESSION.value, random_state=None):
    n_splits = params['n_splits']
    shuffle = params['shuffle']
    # Load dataset
    df = pd.read_csv(csv_file)
    
    # Assume target variable is in the last column
    X = df.iloc[:, :-1].values  # Features
    y = df.iloc[:, -1].values   # Target variable
    
    # Initialize Stratified K-Folds
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    
    # List to store results for each fold
    results = []
    fold_accuracy = []
    
    # Perform Stratified K-Folds split and training
    for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
        # Split the dataset
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Train the model
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        if target_variable_type == ProblemType.CLASSIFICATION.value:
            fold_accuracy.append(accuracy_score(y_test, y_pred))
        elif target_variable_type == ProblemType.REGRESSION.value:
            fold_accuracy.append(mean_absolute_error(y_test, y_pred))
        else:
            raise ValueError("Invalid target_variable_type. Expected 'classification' or 'regression'.")
    
    return CrossValidationResponse('Stratified K-Folds', params, np.mean(fold_accuracy), model)


def repeated_kfold_cv(csv_file, params, model, target_variable_type=ProblemType.REGRESSION.value, random_state=None):
    n_splits = params['n_splits']
    shuffle = params['shuffle']
    n_repeats = params['n_repeats']
    # Load dataset
    df = pd.read_csv(csv_file)
    
    # Assume target variable is in the last column
    X = df.iloc[:, :-1].values  # Features
    y = df.iloc[:, -1].values   # Target variable
    
    # Initialize K-Folds (to be repeated)
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    
    # Prepare list to store the CV splits and results
    results = []
    fold_accuracy = []
    
    # Perform repeated K-Folds
    for repeat in range(n_repeats):
        for fold, (train_index, test_index) in enumerate(kf.split(X, y), 1):
            # Split the dataset
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # Train the model
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            
            if target_variable_type == ProblemType.CLASSIFICATION.value:
                fold_accuracy.append(accuracy_score(y_test, y_pred))
            elif target_variable_type == ProblemType.REGRESSION.value:
                fold_accuracy.append(mean_absolute_error(y_test, y_pred))
            else:
                raise ValueError("Invalid target_variable_type. Expected 'classification' or 'regression'.")
    
    return CrossValidationResponse('Repeated K-Folds', params, np.mean(fold_accuracy), model)


def nested_kfold_split_and_train(csv_file, params, model, target_variable_type=ProblemType.REGRESSION.value, random_state=None):
    n_splits = params['n_splits']
    shuffle = params['shuffle']
    n_repeats = params['n_repeats']
    # Load dataset
    df = pd.read_csv(csv_file)
    
    # Assume target variable is in the last column
    X = df.iloc[:, :-1].values  # Features
    y = df.iloc[:, -1].values   # Target variable
    
    # Initialize outer K-Fold
    outer_kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    
    # Prepare list to store splits and results
    splits = []
    outer_fold_accuracy = []
    
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
            inner_fold_accuracy = []
            for inner_train_index, inner_test_index in inner_kf.split(X_train_outer, y_train_outer):
                # Inner fold splits
                X_train_inner, X_test_inner = X_train_outer[inner_train_index], X_train_outer[inner_test_index]
                y_train_inner, y_test_inner = y_train_outer[inner_train_index], y_train_outer[inner_test_index]
                
                # Train the model on the inner fold
                model.fit(X_train_inner, y_train_inner)
                y_pred_inner = model.predict(X_test_inner)
                
                if target_variable_type == ProblemType.CLASSIFICATION.value:
                    inner_fold_accuracy.append(accuracy_score(y_test_inner, y_pred_inner))
                elif target_variable_type == ProblemType.REGRESSION.value:
                    inner_fold_accuracy.append(mean_absolute_error(y_test_inner, y_pred_inner))
                else:
                    raise ValueError("Invalid target_variable_type. Expected 'classification' or 'regression'.")
                
                inner_splits.append({
                    'X_train': X_train_inner, 'X_test': X_test_inner,
                    'y_train': y_train_inner, 'y_test': y_test_inner,
                    'accuracy': inner_fold_accuracy[-1]
                })
            
            # Train the model on the outer fold
            model.fit(X_train_outer, y_train_outer)
            y_pred_outer = model.predict(X_test_outer)
            
            if target_variable_type == ProblemType.CLASSIFICATION.value:
                outer_fold_accuracy.append(accuracy_score(y_test_outer, y_pred_outer))
            elif target_variable_type == ProblemType.REGRESSION.value:
                outer_fold_accuracy.append(mean_absolute_error(y_test_outer, y_pred_outer))
            else:
                raise ValueError("Invalid target_variable_type. Expected 'classification' or 'regression'.")
            
            # Store the outer fold splits along with the inner fold splits
            splits.append({
                'repeat': repeat + 1,
                'outer_train': (X_train_outer, y_train_outer),
                'outer_test': (X_test_outer, y_test_outer),
                'inner_splits': inner_splits,
                'outer_accuracy': outer_fold_accuracy[-1]
            })
    
    return CrossValidationResponse('Nested K-Folds', params, np.mean(outer_fold_accuracy), model)
