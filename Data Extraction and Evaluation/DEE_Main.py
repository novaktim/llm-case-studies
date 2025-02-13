from ninept import qwen
import subprocess

# Run all the other python files as a process
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

# Output from QWEN LLM
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

    file_path = "kaggle_competitions_details.txt"
    result = llm_evaluation(file_path, selected_dataset)

    print("Hello! This is your QWEN LLM Agent !!!")
    print(result + ". Give me a python code for implementing the evaluation metric?")
    response = qwen(result + ". Give me a python code for implementing the evaluation metric?")
    print(response)
    with open("llm_evaluation_code.py", "w", encoding="utf-8") as file:
        file.write(response)




if __name__ == "__main__":
    run_file("DataExtraction.py")
    run_file("GettingMetadata.py")
    llm_output()
