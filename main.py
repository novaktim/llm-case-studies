import pandas as pd
import numpy as np
import datetime
import re
from ninept import qwen




np.random.seed(42)
n_samples = 500

# Generating synthetic data
age = np.random.randint(18, 80, n_samples)  # Age in years
bmi = np.random.uniform(18.5, 35, n_samples)  # BMI in kg/m²
activity = np.random.uniform(0, 120, n_samples)  # Physical activity in minutes/day
cholesterol = np.random.uniform(150, 280, n_samples)  # Cholesterol level in mg/dL
smoking_status = np.random.choice([0, 1], n_samples)  # 0 = Non-smoker, 1 = Smoker
alcohol_consumption = np.random.uniform(0, 10, n_samples)  # Units of alcohol per week
genetic_risk = np.random.uniform(0, 1, n_samples)  # Genetic risk score (normalized)
height = np.random.uniform(150, 200, n_samples)  # Height in cm
weight = np.random.uniform(50, 120, n_samples)  # Weight in kg

# Generate a random datetime for each entry within the last 10 years
start_date = datetime.datetime.now() - datetime.timedelta(days=365 * 10)
timestamps = [
    (start_date + datetime.timedelta(days=np.random.randint(0, 3650), 
    	seconds=np.random.randint(0, 86400))).isoformat() + 'Z'
    	for _ in range(n_samples)
]

blood_pressure = (
    100
    + 0.4 * age
    + 0.02 * weight
    - 0.01 * height
    - 0.2 * activity
    + 0.3 * cholesterol
    + 5 * smoking_status
    + 2 * alcohol_consumption
    + 10 * genetic_risk
    + np.random.normal(0, 10, n_samples)  # Adding noise
)

medical_data = pd.DataFrame({
    "Age": age,
    "Height": height,
    "Weight": weight,
    "Activity": activity,
    "Cholesterol": cholesterol,
    "SmokingStatus": smoking_status,
    "AlcoholConsumption": alcohol_consumption,
    "GeneticRisk": genetic_risk,
    "BloodPressure": blood_pressure
})

#print("How are you?")
#print(qwen("How are you?"))
#print(qwen("How are you?", role="You are a python program"))
query = "Apply feature engineering to the pandas dataset \"medical_data\" and assume it is already given as a variable and return only and only python code without comments to engineer the new variables: " + medical_data.head().to_string()
print(query + "\n")
#print(qwen(query))

def extract_code_or_keep(input_string):
    # Regular expression to capture Python code within triple backticks
    code_blocks = re.findall(r"```python\n(.*?)```", input_string, re.DOTALL)
    if code_blocks:
        return "\n".join(code_blocks)
    else:
        return input_string
    
def ask_llm_python(medical_data, query, role, tries=3):
    output = qwen(query, role)
    code = extract_code_or_keep(output)
    try:
        print("Trying to execute python code: \n" + code)
        exec(code)
        return medical_data
    except:
        if tries == 0:
            raise Exception(f"Failed to get a valid response from the llm: {query}")
        else:
            return ask_llm_python(medical_data, query + " The last answer was not a valid python code. Please answer only in python code and no explanations or comments.",
            role=role, tries=tries-1)

medical_data2 = ask_llm_python(medical_data, query, role="You are a python program", tries=3)
print("\nFinished\n")
print(medical_data.head().to_string())
#### Vince trial and error stuff