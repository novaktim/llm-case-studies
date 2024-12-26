import pandas as pd
from helper.prompt import generate_qwen_prompt_with_values
from helper.qwen import get_cross_validation_technique
from helper.CV_Executor import perform_cross_validation
import json

train_size, test_size = 60, 20

#Model name and description in your own words
model='decision trees'

#Target variable type: {"timeseries","categorical", "numerical"}
target_variable_type = 'categorical'

#Features type: {"numerical", "categorical","numerical+categorical"}
feature_type = 'categorical'

qwen_response = get_cross_validation_technique(
    generate_qwen_prompt_with_values(
        model,
        target_variable_type,
        feature_type))


cross_validation = json.loads(qwen_response)
result_data = perform_cross_validation(cross_validation)
print(result_data)

