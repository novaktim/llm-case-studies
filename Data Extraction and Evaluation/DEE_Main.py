from ninept import qwen
import pandas as pd
import subprocess
import os

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
    print("\nTarget Variable: " + target_variable.name + "\n")

    file_path = "kaggle_competitions_details.txt"
    result = llm_evaluation(file_path, selected_dataset)

    print("Hello! This is your QWEN LLM Agent !!!")
    print(result + ". Give me a function named evaluation for implementing the evaluation metric?")
    response = qwen(result + ". Give me a function named evaluation for implementing the evaluation metric?")
    print(response)
    with open("llm_evaluation_code.py", "w", encoding="utf-8") as file:
        file.write(response)

if __name__ == "__main__":
    # run_file("DataExtraction.py")
    # run_file("GettingMetadata.py")
    llm_output()
