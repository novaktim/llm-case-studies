import json
import numpy as np
import pandas as pd
from ninept import qwen
import re
import PyPDF2  # For extracting text from PDF

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.
    """
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
    except Exception as e:
        print(f"❌ Error extracting text from PDF: {e}")
        return None

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

# Function to identify the best model using Qwen and the PDF guide
def identify_best_model(X_train, y_train, pdf_context):
    """
    Ask Qwen to identify the best model based on the provided data, target, and PDF guide.
    """
    prompt = (
        "Based on the following dataset and the provided model selection guide, identify the best machine learning model for the given target. "
        "The dataset has the following shapes:\n"
        f"- X_train: {X_train.shape}\n"
        f"- y_train: {y_train.shape}\n"
        "The target is a binary classification problem. "
        "Choose the best model from the following candidates: Logistic Regression, Decision Tree, Random Forest. "
        "Return only the name of the best model (e.g., 'Random Forest')."
    )
    response = qwen(content=pdf_context + "\n\n" + prompt)
    return response.strip()

# Function to generate script for the best model
def generate_best_model_script(best_model_name, X_train, y_train, pdf_context):
    """
    Ask Qwen to generate a script for the best model.
    """
    prompt = (
        f"Generate a complete and executable Python script for the best model ({best_model_name}). "
        "The script should include the following steps:\n"
        "1. Load the preprocessed data (X_train, y_train).\n"
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
    response = query_qwen_for_analysis(prompt, context=pdf_context)
    return extract_model_code(response)

# Function to save the generated script to a JSON file
def save_script_to_json(script_content, json_file_path):
    """
    Save the generated script to a JSON file.
    """
    script_data = {"script": script_content}
    with open(json_file_path, "w") as file:
        json.dump(script_data, file, indent=4)
    print(f"✅ Script saved to JSON file: {json_file_path}")

# Main function
def main():
    # Example preprocessed data (replace with your actual data)
    X_train = np.random.rand(100, 10)  # Example training features
    y_train = np.random.randint(0, 2, 100)  # Example training labels (binary classification)

    # Step 1: Extract text from the PDF guide
    pdf_path = "Model_selection_guide.pdf"  # Replace with the path to your PDF
    pdf_context = extract_text_from_pdf(pdf_path)
    if not pdf_context:
        print("❌ Failed to extract text from the PDF guide.")
        return

    # Step 2: Identify the best model using Qwen and the PDF guide
    best_model_name = identify_best_model(X_train, y_train, pdf_context)
    print(f"🏆 Best Model Identified: {best_model_name}")

    # Step 3: Generate a script for the best model
    best_model_script = generate_best_model_script(best_model_name, X_train, y_train, pdf_context)
    if best_model_script:
        # Save the best model script to a JSON file
        save_script_to_json(best_model_script, "best_model_script.json")
    else:
        print("❌ Failed to generate the best model script.")

if __name__ == "__main__":
    main()