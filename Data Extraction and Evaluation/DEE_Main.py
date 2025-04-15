from ninept import qwen
import pandas as pd
import subprocess
import os
import re

def run_file(filename):
    try:
        print(f"\nRunning {filename}...")
        subprocess.run(["python", filename], check=True)
        print(f"{filename} executed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error executing {filename}: {e}")

def llm_evaluation(file_path, dataset_name):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()

            if dataset_name.lower() in line.lower():
                return line

    return "Dataset description not found."

def llm_output():
    datasets = [
    "digit-recognizer",
    "equity-post-HCT-survival-predictions",
    "home-data-for-ml-course",
    "house-prices-advanced-regression-techniques",
    "spaceship-titanic",
    "store-sales-time-series-forecasting"
    ]

    print("Available datasets:")
    for dataset in datasets:
        print(f"- {dataset}")

    selected_dataset = input("\nEnter the dataset name: ").strip()

    new_file_path = os.path.join("kaggle_datasets", selected_dataset, "train.csv")
    dataframe = pd.read_csv(new_file_path)
    target_variable = dataframe.iloc[:, -1]

    file_path = "kaggle_competitions_details.txt"
    result = llm_evaluation(file_path, selected_dataset)

    details = "Have a look at the following details:\n" + result + "\n\nYou have train, test and submit_submission.csv" + "\n\nThe target variable is " + target_variable.name + "\n\nProvide a function named Evaluation for optimized and minimal the evaluation metric of the competition" + "\n\nOutput only the function, nothing else" + "\n\nThe function must be **minimal**, containing only the required code. No comments, no print statements, no explanations."
    
    print(details)
    
    response = qwen(content = "Have a look at the following details:\n\n" + result + 
        "\n\nYou have train, test and submit_submission.csv" + 
        "\n\nThe target variable is " + target_variable.name + 
        "\n\nProvide a function named Evaluation for optimized and minimal evaluation metric of the competition." + 
        "\n\nOutput only the function, nothing else." + 
        "\n\nThe function must be **minimal**, containing only the required code. No comments, no print statements, no explanations." +
        "\n\nExecute the function and give path to train, test and sample_submission.csv files. Also make sure there is no errors while executing."
        , role = "You are a data scientist assistant" 
        )
    
    pattern = r"```(?:python\n)?(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    newresponse = "\n\n".join(match.strip() for match in matches)

    with open("llm_evaluation_code.py", "w", encoding="utf-8") as file:
        file.write(newresponse)

if __name__ == "__main__":
    run_file("DataExtraction.py")
    run_file("GettingMetadata.py")
    llm_output()

