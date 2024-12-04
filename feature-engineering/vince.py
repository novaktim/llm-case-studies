from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
from ninept import qwen

## Additional information comes as imput from other group

addInfo = ""
dataSummary = ""
colDescription = ""


### impute a dataframe n_imputations many times and return dataframe
def impute_mixed_data(df, n_imputations=1):
    """
    Imputes missing values in a DataFrame using:
        - MICE with XGBoost for numeric columns
        - Most frequent value for categorical columns.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame containing missing values.
        n_imputations (int): Number of imputations for multiple imputation. Defaults to 1.
    
    Returns:
        pd.DataFrame: The DataFrame with imputed values aggregated across imputations.
    """
    # Identify columns
    numeric_cols = df.select_dtypes(include=["number"]).columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    ignored_cols = df.columns.difference(numeric_cols.union(categorical_cols))

    # Define imputers
    xgb_estimator = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        early_stopping_rounds=None,
        verbosity=0
    )
    numeric_imputer = IterativeImputer(estimator=xgb_estimator, max_iter=100, random_state=42)
    categorical_imputer = SimpleImputer(strategy="most_frequent")

    # Perform imputations
    imputations = []
    for _ in range(n_imputations):
        df_imputed = df.copy()
        
        # Impute numeric columns
        if not numeric_cols.empty:
            df_imputed[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])
        
        # Impute categorical columns
        if not categorical_cols.empty:
            df_imputed[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])
        
        imputations.append(df_imputed)

    if n_imputations == 1:
        return imputations[0]
    else:
        # Aggregate multiple imputations
        aggregated_df = imputations[0].copy()
        
        # Average numeric columns
        for col in numeric_cols:
            aggregated_df[col] = np.mean(
                [imputed_df[col].astype(float) for imputed_df in imputations], axis=0
            )
        
        # Mode for categorical columns
        for col in categorical_cols:
            aggregated_df[col] = pd.concat(
                [imputed_df[col] for imputed_df in imputations], axis=1
            ).mode(axis=1)[0]
        
        return aggregated_df




def delete_values(df, p):
    """
    Deletes p percent of all values in the DataFrame by replacing them with NaN.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        p (float): Percentage of values to delete (between 0 and 100).

    Returns:
        pd.DataFrame: The DataFrame with missing values introduced.
    """
    if not (0 <= p <= 100):
        raise ValueError("Percentage 'p' must be between 0 and 100.")

    # Copy the DataFrame to avoid modifying the original
    df = df.copy()

    # Calculate the number of values to replace
    total_values = df.size
    n_missing = int(total_values * (p / 100))

    # Flatten the DataFrame index for easier random sampling
    flat_indices = [(i, j) for i in range(df.shape[0]) for j in range(df.shape[1])]

    # Randomly select indices to replace with NaN
    missing_indices = np.random.choice(len(flat_indices), n_missing, replace=False)

    # Replace the selected values with NaN
    for idx in missing_indices:
        i, j = flat_indices[idx]
        df.iat[i, j] = np.nan

    return df

def delete_values_with_exclusion(df, p, exclude_column):
    """
    Deletes p percent of all values in the DataFrame by replacing them with NaN,
    while ensuring that the specified column does not get any missing values.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        p (float): Percentage of values to delete (between 0 and 100).
        exclude_column (str): Name of the column to exclude from missing values.

    Returns:
        pd.DataFrame: The DataFrame with missing values introduced.
    """
    if not (0 <= p <= 100):
        raise ValueError("Percentage 'p' must be between 0 and 100.")
    if exclude_column not in df.columns:
        raise ValueError(f"Column '{exclude_column}' not found in the DataFrame.")

    # Copy the DataFrame to avoid modifying the original
    df = df.copy()

    # Calculate the number of values to replace
    total_values = df.size - len(df[exclude_column])  # Exclude the protected column
    n_missing = int(total_values * (p / 100))

    # Flatten the DataFrame index for easier random sampling
    flat_indices = [
        (i, j) for i in range(df.shape[0]) for j in range(df.shape[1])
        if df.columns[j] != exclude_column
    ]

    # Randomly select indices to replace with NaN
    missing_indices = np.random.choice(len(flat_indices), n_missing, replace=False)

    # Replace the selected values with NaN
    for idx in missing_indices:
        i, j = flat_indices[idx]
        df.iat[i, j] = np.nan

    return df


# Load the CSV file
file_path = 'mtcars.csv'  # Replace with the actual path to the file
mtcars_df = pd.read_csv(file_path)

mtcars_df = mtcars_df.drop(mtcars_df.columns[0], axis=1)

# original data
print(mtcars_df.head())

#ampute
mtcars_df_mis = delete_values_with_exclusion(mtcars_df, 10, "mpg")
print(mtcars_df_mis.head())

#impute
mtcars_df_imp = impute_mixed_data(mtcars_df_mis, 1)
print(mtcars_df_imp.head())

#complete cases
complete_cases_df = mtcars_df_mis.dropna()

#### compare with and without

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error, accuracy_score

def train_and_compare(df1, df2, response_variable):
    """
    Trains Random Forest models on two datasets and compares their cross-validated performance.

    Parameters:
        df1 (pd.DataFrame): The first dataset.
        df2 (pd.DataFrame): The second dataset.
        response_variable (str): The name of the response variable (target column).

    Returns:
        None: Prints the performance metrics to the screen.
    """
    # Ensure the response variable exists in both datasets
    if response_variable not in df1.columns or response_variable not in df2.columns:
        raise ValueError(f"Response variable '{response_variable}' not found in both datasets.")

    # Determine if the response is numeric (regression) or categorical (classification)
    is_numeric_response = np.issubdtype(df1[response_variable].dtype, np.number)

    # Split features (X) and response variable (y)
    X1, y1 = df1.drop(columns=[response_variable]), df1[response_variable]
    X2, y2 = df2.drop(columns=[response_variable]), df2[response_variable]

    if is_numeric_response:
        # Regression
        model = RandomForestRegressor(random_state=42)
        scorer = make_scorer(mean_squared_error, greater_is_better=False)  # Use MSE as the metric
    else:
        # Classification
        model = RandomForestClassifier(random_state=42)
        scorer = make_scorer(accuracy_score)  # Use accuracy as the metric

    # Cross-validate on both datasets
    cv_scores1 = cross_val_score(model, X1, y1, cv=5, scoring=scorer)
    cv_scores2 = cross_val_score(model, X2, y2, cv=5, scoring=scorer)

    if is_numeric_response:
        # Print MSE (negate scores for MSE since we used greater_is_better=False)
        print(f"Dataset 1 - Mean Squared Error: {abs(cv_scores1.mean()):.4f}")
        print(f"Dataset 2 - Mean Squared Error: {abs(cv_scores2.mean()):.4f}")
    else:
        # Print misclassification (1 - accuracy)
        print(f"Dataset 1 - Misclassification Rate: {1 - cv_scores1.mean():.4f}")
        print(f"Dataset 2 - Misclassification Rate: {1 - cv_scores2.mean():.4f}")


#train_and_compare(mtcars_df_mis, mtcars_df_imp, "mpg")

def add_missingness_correlation_vars(df, response, threshold):
    """
    Adds missingness indicators for columns with missing values based on correlation with the response variable.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        response (str): The name of the response variable column.
        threshold (float): The correlation threshold to add the missingness indicator.

    Returns:
        pd.DataFrame: The modified DataFrame with new columns for significant missingness indicators.
    """
    if response not in df.columns:
        raise ValueError(f"Response variable '{response}' not found in the DataFrame.")
    if not isinstance(threshold, (int, float)):
        raise ValueError("Threshold must be a numeric value.")
    
    # Copy the DataFrame to avoid modifying the original
    df = df.copy()

    # Ensure response is numeric (correlation requires numeric data)
    if not np.issubdtype(df[response].dtype, np.number):
        raise ValueError("The response variable must be numeric to calculate correlations.")

    # Identify columns with missing values
    missingness_cols = [col for col in df.columns if col != response and df[col].isnull().any()]

    for col in missingness_cols:
        # Create a missingness indicator (1 if missing, 0 otherwise)
        missing_indicator = df[col].isnull().astype(int)
        # Calculate correlation with the response variable
        correlation = missing_indicator.corr(df[response])
        
        # Add the indicator as a new column if correlation exceeds the threshold
        if abs(correlation) > threshold:
            new_col_name = f"{col}_missing"
            df[new_col_name] = missing_indicator

    return df




mtcars_inclMissingIndicator = add_missingness_correlation_vars(mtcars_df_mis, "mpg", 0.1)
mtcars_inclMissingIndicator = add_missingness_correlation_vars(mtcars_df_mis, "mpg", 0.1)
mtcars_inclMissingIndicator = impute_mixed_data(mtcars_inclMissingIndicator, 1)
print(mtcars_inclMissingIndicator.head(10))

train_and_compare(mtcars_df, mtcars_df_imp, "mpg")
train_and_compare(mtcars_df, complete_cases_df, "mpg")
train_and_compare(mtcars_df, mtcars_inclMissingIndicator, "mpg")


#TODO compare with single imputation
#implement missing as features (only makes sense for non MCAR)
# early stopping not reached
# how to impute categorical?



# Ask Qwen which missingness structure could have an influence of the response
colnames_string = ", ".join(mtcars_df.columns)

query = "Have a look at the following columns: " + colnames_string + " . Also consider the dataframe description: " + dataSummary + " , the description of the columns: " + colDescription + " these additional information: " + addInfo +  " and try to have an educated guess, for which variable the indicator whether the value is missing or not could have predictive power on the response variable: mpg. Only output a single column index for which you think the column is the most relevant for predicting mpg, do not output anything else!"


print(query)
#print(qwen(query))


# Imagine the response has to be a number
def read(output):
    output = output.strip()
    output.replace(",", ".")
    return int(output)

def call_llm(content, role, tries=10):
    outp = qwen(content, role)
    try:
        return read(outp)
    except:
        if tries == 0:
            raise Exception("Failed to get a valid response from the llm (" + str(outp) + ")")
        else:
            return call_llm(content + f"The last string ('{outp}') was not a valid number. Please answer only with an integer number", role, tries - 1)
        
print(call_llm(query, "data science expert"))

#TODO 
#return array of ints und nicht einzelnen int
#print number of tries needed

#paar comments yur praesi
#naechste 2 wochen nach naechster frei
#built pipeline with LLM feature engeeneering and imputation and compare results

#ask Gwen which features should be used as polynomial or log transformed features
#Binning, Interaction features
#Handling Temporal Features
#Extract Date Components: Extract day, month, year, or day of the week from timestamps.
#Lag Features: Create features based on past values (useful in time-series data).
#Rolling Statistics: Compute rolling means, sums, or standard deviations over time.

#exclude indicator features and constant features by hand

#also shoul indicate in query that it has the option to not return anything


# Imagine the response has to be an array of integers
def read_mv(output):
    output = output.strip()
    output = output.replace(",", ".")
    # Split the output into parts and try converting each to an integer
    return [int(value) for value in output.split()]

def call_llm_mv(content, role, tries=10):
    outp = qwen(content, role)
    try:
        return read_mv(outp)
    except:
        if tries == 0:
            raise Exception("Failed to get a valid response from the llm (" + str(outp) + ")")
        else:
            print("This try did not work: " + str(tries))
            return call_llm_mv(
                content + f"The last string ('{outp}') was not a valid array of integers. Please answer only with a space-separated list of integers.",
                role,
                tries - 1
            )
query_mv = "Have a look at the following columns: " + colnames_string + " . Also consider the dataframe description: " + dataSummary + " , the description of the columns: " + colDescription + " these additional information: " + addInfo +  " and try to have an educated guess, for which variable the indicator whether the value is missing or not could have predictive power on the response variable: mpg. Only output the column indices for which you think the column is relevant for predicting mpg, so return a list of integers and do not output anything else!"
print(query_mv)
print(call_llm_mv(query_mv, "data science expert"))