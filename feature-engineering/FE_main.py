#### This is the function supposed to be called in the overall pipeline ######

import sys
sys.path.append('/feature-engineering')
#import vince
import feature_generation
#1 call imputation
#2 call vince FE approach
#3 call Tim FE approach
#4 aggregate and return


def fe_main(df, eda_summary, ext_info, response): 
    df_new = vince_feature_engineering(df, eda_summary, ext_info, response)
    df_new = tim_feature_engineering(df_new, eda_summary, ext_info, response)
    
    return df_new


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


#TODO
#implementation summary return for model selection
#description, what feature engineering what was done
#description what we did with missing values
    
