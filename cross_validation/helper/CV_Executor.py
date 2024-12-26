from classes.CV import CrossValidation
from dotenv import load_dotenv
import os

load_dotenv()


data_path =  os.path.join(
    os.getenv('DATASET_DIRECTORY'),
    os.getenv('DATA_FILE_NAME')
)
def perform_cross_validation(cv_method_object):
    print(cv_method_object)
    """
    Calls the corresponding cross-validation function based on the method number.

    Parameters:
    - cv_method_number (int): The number representing the cross-validation method.

    Returns:
    - The output of the cross-validation function.
    """
    for cv_method in CrossValidation:
        if cv_method.number == cv_method_object['cv']:
            return cv_method.execute(data_path, params = cv_method_object)
        
    raise ValueError(f"Invalid cross-validation method number: {cv_method_object['cv']}")