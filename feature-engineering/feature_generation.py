import re
from ninept import qwen



def extract_code_or_keep(input_string):
    """
    Extract Python code blocks from a string or return the original string if no code is found.

    This function searches for Python code enclosed within triple backticks (```) and extracts it. 
    If no such block is found, it returns the input string as-is.

    Parameters:
        input_string (str): The input text that may contain Python code blocks.

    Returns:
        str: The extracted Python code if found; otherwise, the original input string.
    
    Example Usage:
        >>> text = "Here is some code:\n```python\nprint('Hello World')\n```"
        >>> extract_code_or_keep(text)
        "print('Hello World')"

        >>> extract_code_or_keep("No code here.")
        "No code here."
    """
    
    # Regular expression to capture Python code within triple backticks
    code_blocks = re.findall(r"```python\n(.*?)```", input_string, re.DOTALL)
    if code_blocks:
        return "\n".join(code_blocks)
    else:
        # Happens if the answer of the LLM is cut off before finishing the code block
        match = re.search(r"```python\n(.*?)(```|$)", input_string, re.DOTALL)
        if match:
            python_code = match.group(1).strip()
            # Remove last line that might be incomplete
            lines = python_code.split("\n")
            if len(lines) > 1:
                python_code = "\n".join(lines[:-1])
            else:
                python_code = ""
            return python_code
        
        return input_string
        

def ask_llm_python(query, role, tries=3):
    """
    Interact with a language model to execute generated Python code on a dataset.

    This function sends a query to an LLM (language model), extracts any returned Python 
    code using `extract_code_or_keep`, and attempts to execute the extracted code. If 
    execution fails, it retries with an adjusted query up to a specified number of attempts.

    Parameters:
        query (str): The prompt sent to the LLM describing what needs to be done.
        role (str): The role/personality of the LLM during interaction.
        tries (int): The maximum number of retries if valid Python output is not received. Default is 3.

    Returns:
        tuple: A tuple containing the possibly transformed dataset and the executed Python code.

    Raises:
        Exception: If all retry attempts fail to produce valid executable Python output.

    Example Usage:
        >>> ask_llm_python(my_dataset, "Generate new features for this dataset.", role="You are a data scientist", tries=2)
    """

    try:
        output = qwen(query, role)
        return output
    except:
        if tries == 0:
            raise Exception(f"Failed to get a valid response from the LLM: {query}")
        else:
            return ask_llm_python(query + " The last answer was not a valid python code. Please answer only in python code without explanations or comments.",
            role=role, tries=tries-1)
            


def feature_generation(original_dataset, eda_summary="", ext_info="", response=""):
    """
    Perform feature engineering on a pandas DataFrame using an LLM-generated transformation script.

    This function interacts with a language model to generate new features or modify existing ones in the given dataset. 
    It builds a query based on the dataset's structure, exploratory data analysis (EDA) summary, and additional context 
    provided by the user. The resulting transformations are applied directly to a copy of the original dataset.

    Parameters:
        original_dataset (pandas.DataFrame): The input dataset that will undergo feature engineering.
        eda_summary (str, optional): A textual description summarizing insights from exploratory data analysis.
                                      If too long (>2000 characters), it is truncated. Defaults to "".
        ext_info (str, optional): Additional contextual information about the data that may aid in feature generation.
                                  If too long (>2000 characters), it is truncated. Defaults to "".
        response (str, optional): Reserved for potential future use or feedback from previous steps. Defaults to "".

    Returns:
        tuple: A tuple containing:
            - transformed_dataset (pandas.DataFrame): The modified dataset after applying feature engineering.
            - generation_summary (str): A summary of changes made by the LLM-generated transformation script.
    """
    
    # Copy dataset in case an error happens
    dataset = original_dataset.copy()



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
    query = query + ("Assume \"dataset\" is already given as a variable. Dont drop any features. Return only python code "
            "to derive new interesting features for machine learning: " 
            + header.to_string()
    )
    # print("Asking gwen:", query + "\n")
    try:
        output = ask_llm_python(query, role="You are a python program", tries=3)
        
        with open("feature_generation_llm_output.txt", "w") as log:
            log.write(f"Qwen generated the following python code to transform the dataset:\n{output}")
            
        exec_code = extract_code_or_keep(output)
        
        #exec(exec_code)
        exec_env = {"dataset": dataset}
        exec(exec_code, exec_env)
        dataset = exec_env["dataset"]
        
        # Ask gwen what changes to the dataset were made.
        query = "Write a summary of the features that were generated or changed by this code: " + exec_code
        # print("Asking gwen:", query + "\n")
        generation_summary = qwen(query)
    

    except Exception as e:
        dataset = original_dataset
        generation_summary = f"No features have been flexibly generated."
        with open("error_feature_generation.txt", "a") as log:
            log.write(f"Error that happened: {e}")

    # Output what transformations were made by the LLM
    # print(generation_summary)
    # print("\nFinished with feature generation.\n")
    return dataset, generation_summary
    

    
