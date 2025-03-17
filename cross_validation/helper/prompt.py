from ninept import qwen

def generate_qwen_prompt_with_values(model, target_variable_type, feature_type):
    cross_validation_mapping = {
        1: "Hold-Out Cross-Validation",
        2: "K-Fold Cross-Validation",
        3: "Leave-One-Out Cross-Validation (LOOCV)",
        4: "Leave-P-Out Cross-Validation",
        5: "Stratified K-Fold Cross-Validation",
        6: "Repeated K-Fold Cross-Validation",
        7: "Nested K-Fold Cross-Validation",
        8: "Rolling Window Partition",
    }

    # Mapping of cross-validation methods and their attributes with expected values
    cv_attributes_mapping = {
        1: {"test_size": 0.2},
        2: {"n_splits": 5},
        3: {},
        4: {"p": 2},
        5: {
            "n_splits": 5,
            "shuffle": True,
        },
        6: {
            "n_splits": 5,
            "n_repeats": 10,
            "shuffle": True,
        },
        7: {
            "n_splits": 5,
        },
        8: {},
    }

    # Generate the prompt
    prompt = f""" 
        Given the following context, choose a number between 1 and 8, where each number corresponds to a unique cross-validation technique. 
        Your task is to suggest the most suitable cross-validation method based on the provided details:

        - **Model**: {model}  
        - **Target Variable Type**: {target_variable_type}  
        - **Feature Type**: {feature_type}  

        Here is the mapping of numbers to cross-validation techniques:
        {cross_validation_mapping}

        Please choose the number that best represents your choice of cross-validation technique relevant for this partcular data and model type.
        
        After selecting the number, also provide values for all the attribute mapped to 
        that selected cross validation technique, as a JSON object. For example:
        
        {{ "cv": 1, "test_size": 0.2 }}
        
        Result should include both the number and the corresponding attributes with values in the form of a JSON object.
        
        Just a json and nothing else.
    """

    # Return the generated prompt
    return prompt





