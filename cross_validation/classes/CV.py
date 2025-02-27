from enum import Enum
from helper.classifications_cv import (
    hold_out_split_and_train,
    get_k_folds_splits_and_train,
    get_leave_one_out_splits_and_train,
    get_leave_p_out_splits_and_train,
    stratified_kfold_cv,
    repeated_kfold_cv,
    nested_kfold_split_and_train,
)
from helper.time_series_cv import rolling_window_partition_df
from dotenv import load_dotenv
import os

load_dotenv()


data_path =  os.path.join(
    os.getenv('DATASET_DIRECTORY'),
    os.getenv('DATA_FILE_NAME')
)

class CrossValidation(Enum):
    HOLD_OUT = (1, hold_out_split_and_train)
    K_FOLD = (2, get_k_folds_splits_and_train)
    LOOCV = (3, get_leave_one_out_splits_and_train)
    LPOCV = (4, get_leave_p_out_splits_and_train)
    STRATIFIED_KFOLD = (5, stratified_kfold_cv)
    REPEATED_KFOLD = (6, repeated_kfold_cv)
    NESTED_KFOLD = (7, nested_kfold_split_and_train)
    ROLLING_WINDOW = (8, rolling_window_partition_df)

    def __init__(self, number, func):
        self.number = number
        self.func = func

    def execute(self, *args, **kwargs):
        """
        Executes the associated function with the provided arguments.
        
        Returns:
        - The output of the associated function.
        """
        return self.func(*args, **kwargs)

class CrossValidationHelper():
    @staticmethod
    def perform_cross_validation(cv_method_object, model, target_variable_type=None):
        """
        Calls the corresponding cross-validation function based on the method number.

        Parameters:
        - cv_method_number (int): The number representing the cross-validation method.
        - model: The model to be evaluated.
        - target_variable_type: The type of the target variable. If None, the target variable is assumed to be continuous.

        Returns:
        - The output of the cross-validation function.
        """
        for cv_method in CrossValidation:
            if cv_method.number == cv_method_object['cv']:
                return cv_method.execute(data_path, params = cv_method_object, model=model, target_variable_type=target_variable_type)
            
        raise ValueError(f"Invalid cross-validation method number: {cv_method_object['cv']}")