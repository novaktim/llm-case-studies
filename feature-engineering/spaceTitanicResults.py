import sys
#from EDA.eda_analysis import *
sys.path.append('/feature-engineering')
from  fe_standard_functions import *
from  FE_main import *
from ninept import qwen


ext = ("The Spaceship Titanic competition on Kaggle is a machine learning challenge that invites " +
"participants to predict which passengers were transported to an alternate dimension during the" +
"spaceship's collision with a spacetime anomaly. This scenario is inspired by the famous Titanic " +
"disaster, reimagined in a futuristic space setting." +
"Participants are tasked with developing a model that can accurately determine whether a "
"passenger was transported to another dimension based on the available data. " +
"The challenge serves as an excellent opportunity for both beginners and experienced data " +
"scientists to apply and enhance their predictive modeling skills.")

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from ninept import qwen

#print(df.head())
#print(df.describe())
def automated_eda_numerical_only(df, unique_threshold=0.7):
    description = ""  # Initialize the description string
    
    # Basic Information
    description += "--- Basic Information ---\n"
    description += f"Number of Rows: {df.shape[0]}\n"
    description += f"Number of Columns: {df.shape[1]}\n"
    description += "\nColumn Names:\n"
    description += ", ".join(df.columns.tolist()) + "\n\n"

    # Data Types
    description += "--- Data Types ---\n"
    description += df.dtypes.to_string() + "\n\n"

    
    # Missing Values Analysis
    description += "\nMissing Values:\n"
    missing_vals = df.isnull().sum()
    missing_data = missing_vals[missing_vals > 0]
    description += str(missing_data) + "\n"
    description += "\nPercentage of Missing Values:\n"
    description += str((missing_vals / len(df)) * 100) + "\n"
    
    # Drop high-uniqueness categorical columns
    high_uniqueness_cols = []
    for col in df.select_dtypes(include=[object]).columns:
        if df[col].nunique() > unique_threshold * len(df):
            high_uniqueness_cols.append(col)
    df = df.drop(columns=high_uniqueness_cols)
    description += f"\nDropped High-Uniqueness Columns: {high_uniqueness_cols}\n"

    # Constant Columns
    constant_columns = [col for col in df.columns if df[col].nunique() == 1]
    description += "--- Constant Columns ---\n"
    if constant_columns:
        description += f"The dataset contains {len(constant_columns)} constant column(s): {', '.join(constant_columns)}\n"
    else:
        description += "No constant columns detected.\n\n"

    # Duplicate Rows
    duplicate_rows = df.duplicated().sum()
    description += "--- Duplicate Rows ---\n"
    description += f"Number of duplicate rows: {duplicate_rows}\n\n"

    # Distribution Analysis for Numeric Columns
    description += "\nSkewness and Kurtosis of Numeric Features:\n"
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    skew_kurt = pd.DataFrame({
        "Skewness": df[numeric_cols].apply(skew),
        "Kurtosis": df[numeric_cols].apply(kurtosis)
    })
    description += str(skew_kurt) + "\n"
    
    # Boxplot Summary Information
    description += "\nBoxplot Summary Information (Min, Q1, Median, Q3, Max):\n"
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        min_val = df[col].min()
        max_val = df[col].max()
        median = df[col].median()
        description += f"{col} - Min: {min_val}, Q1: {q1}, Median: {median}, Q3: {q3}, Max: {max_val}\n"
        
        # Outliers based on IQR method
        outliers = df[(df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))]
        description += f"  Outliers in {col}: {len(outliers)}\n"
    
    # Correlation Analysis on numeric data only
    correlation = df.select_dtypes(include=[np.number]).corr()
    description += "\nCorrelation Analysis (Numeric Feature Pairs):\n"
    for i in range(len(correlation.columns)):
        for j in range(i):
            description += f"{correlation.columns[i]} and {correlation.columns[j]} - Correlation: {correlation.iloc[i, j]:.2f}\n"
    
    # Categorical Columns Analysis
    categorical_cols = df.select_dtypes(include=[object]).columns
    description += "\n--- Categorical Columns ---\n"
    description += f"Categorical Columns: {categorical_cols.tolist()}\n"
    
    description += "\n--- Count Analysis of Categorical Features ---\n"
    for col in categorical_cols:
        unique_values = df[col].nunique()
        most_frequent_value = df[col].mode()[0]
        most_frequent_count = df[col].value_counts().iloc[0]
        description += f"{col} - Unique Values: {unique_values}, Most Frequent Value: {most_frequent_value} ({most_frequent_count} occurrences)\n"
    description += "\nAutomated EDA Completed."
    
    return description, df

file_path = 'data/train.csv'  
titanic_df = pd.read_csv(file_path)

eda, _ = automated_eda_numerical_only(titanic_df)
print(eda)

results = fe_main(titanic_df, response = "Transported", eda_summary = eda, ext_info = ext, 
                  apply_standardization=False)

print("########################### results Titanic Spaceship ################################### \n")
print(results["fe_summary"])
results["df_new"].to_csv("data/spaceTitanicResults.csv", index=False)