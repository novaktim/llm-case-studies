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
            


def feature_generation(original_dataset, eda_summary="", ext_info="", response=""):
    
    # Copy dataset in case an error happens
    transformed_dataset = original_dataset.copy()

    # If the description is too long, shorten it
    if len(eda_summary) > 2000:
        eda_summary = eda_summary[:2000] + "...\n"
    if len(ext_info) > 2000:
        ext_info = ext_info[:2000] + "...\n"

    # Build the query for feature generation including additional information about the dataset    
    query = "Apply feature engineering to the pandas dataset \"dataset\". "
    if eda_summary != "":
         query = query + " The dataset is described like this: " + eda_summary + "\n"
    if ext_info != "":
         query = query + " Here is some additional knowledge about the data: " + ext_info + "\n"
    
    # If the header is too long for the query, shorten the columns
    header = original_dataset.head(3)
    if header.shape[1] > 200:
        header = header.iloc[:, :200]

    # Add the header rows to the query, to describe our dataset
    query = query + ("Assume \"dataset\" is already given as a variable and return only python code "
            "to derive the new interesting variables for machine learning: " 
            + header.to_string()
    )
    print("Asking gwen:", query + "\n")
    try:
        transformed_dataset, code = ask_llm_python(transformed_dataset, query, role="You are a python program", tries=3)

        # Ask gwen what changes to the dataset were made.
        query = "Write a summary of the features that were generated or changed by this code: " + code
        print("Asking gwen:", query + "\n")
        generation_info = qwen(query)

    except:
        transformed_dataset = original_dataset
        generation_info = "No changes to the dataset where made."
    
    # Output what transformations were made by the LLM
    print(generation_info)
    print("\nFinished with feature generation.\n")
    return transformed_dataset, generation_info
    

    