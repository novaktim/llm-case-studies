from enum import Enum
from helper.classifications_cv import (
    hold_out_split,
    get_k_folds_splits,
    get_leave_one_out_splits,
    get_leave_p_out_splits,
    stratified_kfold_cv,
    repeated_kfold_cv,
    nested_kfold_split,
)
from helper.time_series_cv import rolling_window_partition_df

class CrossValidation(Enum):
    HOLD_OUT = (1, hold_out_split)
    K_FOLD = (2, get_k_folds_splits)
    LOOCV = (3, get_leave_one_out_splits)
    LPOCV = (4, get_leave_p_out_splits)
    STRATIFIED_KFOLD = (5, stratified_kfold_cv)
    REPEATED_KFOLD = (6, repeated_kfold_cv)
    NESTED_KFOLD = (7, nested_kfold_split)
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
