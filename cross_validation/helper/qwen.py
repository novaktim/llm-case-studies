from ninept import qwen

def get_cross_validation_technique(prompt):
    return qwen(prompt,role='you are a cross validation expert')