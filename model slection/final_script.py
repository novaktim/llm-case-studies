import json
import re
import numpy as np
import pandas as pd
from ninept import qwen
import subprocess

# Function to extract Python code from Qwen's response
def extract_model_code(response):
    """
    Extract Python code from Qwen's response using triple backticks.
    """
    pattern = r"```python\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    return "\n".join(matches).strip() if matches else None

# Function to query Qwen for analysis
def query_qwen_for_analysis(prompt, context=""):
    """
    Query Qwen for a Python script based on the given prompt and context.
    """
    full_prompt = f"{context}\n\n{prompt}" if context else prompt
    strict_prompt = (
        "Provide a complete and executable Python script for the given request. "
        "Ensure that all necessary imports, functions, and logic are included. "
        "DO NOT use an external dataset URL. Use the provided preprocessed data. "
        "Do NOT truncate responses. If the output is long, return it in sequential blocks using triple backticks ```python```."
    )
    response = qwen(content=full_prompt + "\n\n" + strict_prompt)
    return response

# Function to generate model evaluation script
def generate_model_evaluation_script(X_train, X_test, y_train, y_test):
    """
    Generate a Python script for model evaluation using Qwen.
    """
    prompt = (
        "Generate a complete and executable Python script for cross-validation and model evaluation. "
        "Use the preprocessed dataset (X_train, X_test, y_train, y_test) provided from preprocessing. "
        "The preprocessed data has the following shapes:\n"
        f"- X_train: {X_train.shape}\n"
        f"- X_test: {X_test.shape}\n"
        f"- y_train: {y_train.shape}\n"
        f"- y_test: {y_test.shape}\n"
        "Include the following steps in the script:\n"
        "1. Define evaluation metrics: accuracy, precision, recall, and F1-score.\n"
        "2. Create and configure candidate models (Logistic Regression, Decision Tree, Random Forest).\n"
        "3. Implement cross-validation using `cross_val_score` with 5-fold validation for each model.\n"
        "4. Fit each model on the training data (X_train, y_train) and evaluate on test data (X_test, y_test).\n"
        "5. Compute and store the evaluation metrics for each model.\n"
        "6. Save the results to a CSV file named 'model_evaluation_results.csv'.\n"
        "7. Print the evaluation results for each model.\n"
        "8. Identify the best-performing model based on accuracy.\n"
        "9. Ensure the script is complete, executable, and properly formatted.\n"
        "10. Use consistent indentation (4 spaces per level).\n"
        "11. Ensure all brackets (`[`, `{`, `(`) are properly closed.\n"
        "12. Ensure the script is properly formatted and free of syntax errors.\n"
        "13. Avoid unexpected indentation or misaligned code blocks.\n"
        "14. Generate complete and executable code without truncation or omissions.\n"
        "15. Strictly enforce proper indentation (4 spaces per level) for all code blocks.\n"
        "16. Ensure all nested structures (e.g., lists, dictionaries, loops) are properly indented.\n"
        "17. Explicitly define the `models` list with proper indentation and closure, as shown below:\n"
        "    models = [\n"
        "        ('Logistic Regression', LogisticRegression()),\n"
        "        ('Decision Tree', DecisionTreeClassifier()),\n"
        "        ('Random Forest', RandomForestClassifier())\n"
        "    ]\n"
        "DO NOT use an external dataset URL. Use the preprocessed data provided from Step 2."
    )
    response = query_qwen_for_analysis(prompt)
    return extract_model_code(response)

# Function to generate script for the best model
def generate_best_model_script(best_model_name):
    """
    Generate a Python script for the best model using Qwen.
    """
    prompt = (
        f"Generate a complete and executable Python script for the best model ({best_model_name}). "
        "The script should include the following steps:\n"
        "1. Load the preprocessed data (X_train, X_test, y_train, y_test).\n"
        "2. Initialize the best model with default or recommended hyperparameters.\n"
        "3. Fit the model on the training data (X_train, y_train).\n"
        "4. Save the trained model to a file (e.g., using `joblib` or `pickle`).\n"
        "5. Ensure the script is complete, executable, and properly formatted.\n"
        "6. Use consistent indentation (4 spaces per level).\n"
        "7. Ensure all brackets (`[`, `{`, `(`) are properly closed.\n"
        "8. Ensure the script is properly formatted and free of syntax errors.\n"
        "9. Avoid unexpected indentation or misaligned code blocks.\n"
        "10. Generate complete and executable code without truncation or omissions.\n"
        "11. Strictly enforce proper indentation (4 spaces per level) for all code blocks.\n"
        "12. Ensure all nested structures (e.g., lists, dictionaries, loops) are properly indented.\n"
        "DO NOT use an external dataset URL. Use the provided preprocessed data."
    )
    response = query_qwen_for_analysis(prompt)
    return extract_model_code(response)

# Function to save script to a JSON file
def save_script_to_json(script_content, json_file_path):
    """
    Save the generated script to a JSON file.
    """
    script_data = {"script": script_content}
    with open(json_file_path, "w") as file:
        json.dump(script_data, file, indent=4)
    print(f"✅ Script saved to {json_file_path}")

# Function to execute a Python script
def execute_script(script_content):
    """
    Execute a Python script and capture its output.
    """
    # Save script to a temporary file
    temp_script_path = "temp_executable_script.py"
    with open(temp_script_path, "w") as file:
        file.write(script_content)

    # Execute the script
    result = subprocess.run(["python", temp_script_path], capture_output=True, text=True)
    print("\n✅ Execution Output:\n", result.stdout)
    if result.stderr:
        print("\n⚠️ Execution Errors:\n", result.stderr)
    return result

# Example usage
if __name__ == "__main__":
    # Example preprocessed data (replace with your actual data)
    X_train = np.random.rand(100, 10)
    X_test = np.random.rand(50, 10)
    y_train = np.random.randint(0, 2, 100)
    y_test = np.random.randint(0, 2, 50)

    # Step 1: Generate and execute the model evaluation script
    evaluation_script = generate_model_evaluation_script(X_train, X_test, y_train, y_test)
    if evaluation_script:
        # Save the evaluation script to a JSON file
        save_script_to_json(evaluation_script, "model_evaluation_script.json")

        # Execute the evaluation script
        print("🔄 Executing model evaluation script...")
        evaluation_result = execute_script(evaluation_script)

        # Step 2: Identify the best model from the evaluation results
        if not evaluation_result.stderr:  # Ensure no errors occurred
            results_df = pd.read_csv("model_evaluation_results.csv")
            best_model_name = results_df.loc[results_df["Accuracy"].idxmax(), "Model"]
            print(f"\n🏆 Best Model: {best_model_name}")

            # Step 3: Generate the script for the best model
            best_model_script = generate_best_model_script(best_model_name)
            if best_model_script:
                # Save the best model script to a JSON file
                save_script_to_json(best_model_script, "best_model_script.json")
            else:
                print("❌ Failed to generate the best model script.")
        else:
            print("❌ Model evaluation script execution failed.")
    else:
        print("❌ Failed to generate the model evaluation script.")