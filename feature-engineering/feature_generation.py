import pandas as pd
import numpy as np
import datetime
import re
from ninept import qwen


def extract_code_or_keep(input_string):
        # Regular expression to capture Python code within triple backticks
        code_blocks = re.findall(r"```python\n(.*?)```", input_string, re.DOTALL)
        if code_blocks:
            return "\n".join(code_blocks)
        else:
            return input_string
        

def ask_llm_python(dataset, query, role, tries=3):
        output = qwen(query, role)
        code = extract_code_or_keep(output)
        try:
            print("Trying to execute python code: \n" + code)
            exec(code)
            return dataset
        except:
            if tries == 0:
                raise Exception(f"Failed to get a valid response from the llm: {query}")
            else:
                return ask_llm_python(dataset, query + " The last answer was not a valid python code. Please answer only in python code without explanations or comments.",
                role=role, tries=tries-1)
            

def ask_llm_changes(dataset, transformedDataset, query, role, tries=3):
        output = qwen(query, role)
        code = extract_code_or_keep(output)
        try:
            print("Trying to execute python code: \n" + code)
            exec(code)
            return dataset
        except:
            if tries == 0:
                raise Exception(f"Failed to get a valid response from the llm: {query}")
            else:
                return ask_llm_python(dataset, query + " The last answer was not a valid python code. Please answer only in python code without explanations or comments.",
                role=role, tries=tries-1)


def feature_generation(dataset):
    
    query = "Apply feature engineering to the pandas dataset \"dataset\" and assume it is already given as a variable and return only and only python code without comments to engineer the new variables: " + medical_data.head().to_string()
    print("Asking gwen: ", query + "\n")

    transformedDataset = ask_llm_python(dataset, query, role="You are a python program", tries=3)
    print("\nFinished\n")
    print(dataset.head().to_string())

    # Ask gwen what changes to the dataset were made.
    output = qwen(query)

    