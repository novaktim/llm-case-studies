class CrossValidationResponse():
    def __init__(self, method, params, mean_accuracy, trained_model):
        """
        Initializes the CrossValidationResponse object.

        Parameters:
        - method (str): The name of the cross-validation method used.
        - results (any): The results obtained from the cross-validation.
        - params (dict): The parameters used for the cross-validation.
        - mean_accuracy (float): The mean accuracy obtained from the cross-validation.
        - trained_model (any): The trained model obtained from the cross-validation.
        """
        self.method = method
        self.params = params
        self.mean_accuracy = mean_accuracy
        self.trained_model = trained_model

    def to_dict(self):
        """
        Converts the CrossValidationResponse object to a dictionary.

        Returns:
        - dict: A dictionary representation of the CrossValidationResponse object.
        """
        return {
            'method': self.method,
            'params': self.params,
            'mean_accuracy': self.mean_accuracy,
            'trained_model': self.trained_model
        }
        
    def __str__(self):
        """
        Returns a string representation of the CrossValidationResponse object.

        Returns:
        - str: A string representation of the CrossValidationResponse object.
        """
        return (f"CrossValidationResponse(method={self.method}, "
                f"params={self.params}, mean_accuracy={self.mean_accuracy}, "
                f"trained_model={self.trained_model})")