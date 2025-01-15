import re
from ninept import qwen


"""
Documentation
"""
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
            exec(code)
            print("Sucessfully executed the python code: \n" + code)
            return dataset, code
        except:
            if tries == 0:
                raise Exception(f"Failed to get a valid response from the llm: {query}")
            else:
                return ask_llm_python(dataset, query + " The last answer was not a valid python code. Please answer only in python code without explanations or comments.",
                role=role, tries=tries-1)
            


def feature_generation(dataset):
    
    # Copy dataset in case an error happens
    transformedDataset = dataset.copy()
    
    query = ("Apply feature engineering to the pandas dataset \"dataset\" "
            "and assume it is already given as a variable and return only python code "
            "without comments to derive the new interesting variables for machine learning: " 
            + dataset.head(3).to_string()
    )
    print("Asking gwen:", query + "\n")
    try:
        transformedDataset, code = ask_llm_python(transformedDataset, query, role="You are a python program", tries=3)

        # Ask gwen what changes to the dataset were made.
        query = "Write a summary of the features that were generated or changed by this code: " + code
        print("Asking gwen:", query + "\n")
        summary = qwen(query)

    except:
        transformedDataset = dataset
        summary = "No changes to the dataset where made."
    
    # Output what transformations were made by the LLM
    print(summary)
    print("\nFinished with feature generation.\n")
    return transformedDataset, summary
    

    