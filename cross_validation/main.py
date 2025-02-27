import pandas as pd
from helper.prompt import generate_qwen_prompt_with_values
from helper.qwen import get_cross_validation_technique
from helper.datatype_extraction import get_feature_data_types, get_target_data_types
from classes.CV import CrossValidationHelper
import json
from dotenv import load_dotenv
import os
from sklearn.linear_model import LinearRegression

load_dotenv()

trained_model = LinearRegression()

#Model name and description in your own words
model='decision trees'

target_variable_type = get_target_data_types(os.getenv('DATASET_DIRECTORY') + os.getenv('DATA_FILE_NAME'), "target_column_name")

feature_type = get_feature_data_types(os.getenv('DATASET_DIRECTORY') + os.getenv('DATA_FILE_NAME'))

qwen_response = get_cross_validation_technique(
    generate_qwen_prompt_with_values(
        model,
        target_variable_type,
        feature_type))

cross_validation = json.loads(qwen_response)

#to perform cross validation based on the qwen response
result_data = CrossValidationHelper.perform_cross_validation(cross_validation, trained_model)

#resulting data would be the cross validated data based on the technique selected by qwen
#you can use this data to train any of the models you want to train

#Bagging and boosting classes can also be used to train advanced models
#you can use the classes in the helper folder to train the models
