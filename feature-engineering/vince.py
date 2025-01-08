from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
from ninept import qwen
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error, accuracy_score

############################################ Data ##########################################


# Additional information comes as imput from other group
addInfo = ""
dataSummary = ""
colDescription = ""
#description as dictionary?
responseVar = "mpg" #testing data
# Load the CSV file
file_path = 'mtcars.csv'  
mtcars_df = pd.read_csv(file_path)

mtcars_df = mtcars_df.drop(mtcars_df.columns[0], axis=1)
colnames_string = ", ".join(mtcars_df.columns)
# original data
#print(mtcars_df.head())


######################################## Lots of Functions ####################################

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


#query = "Have a look at the following columns: " + colnames_string + " . Also consider the dataframe description: " + dataSummary + " , the description of the columns: " + colDescription + " these additional information: " + addInfo +  " and try to have an educated guess, for which variable the indicator whether the value is missing or not could have predictive power on the response variable: " + responseVar + ". Only output a single column index for which you think the column is the most relevant for predicting:" + responseVar + ", do not output anything else!"
#print(query)
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
        

#print(call_llm(query, "data science expert"))

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

#function to add dummy missingness to the dataframe for all columns in indices
def add_missingness_columns(df, indices):
    """
    Adds missingness indicators for columns with missing values based on correlation with the response variable.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        indices (list of ints): column indices to add missingness dummy columns

    Returns:
        pd.DataFrame: The modified DataFrame with new columns
    """
    # Copy the DataFrame to avoid modifying the original
    df = df.copy()

    for col in indices:
        # Create a missingness indicator (1 if missing, 0 otherwise)
        missing_indicator = df[col].isnull().astype(int)

        if index < 0 or index >= len(df.columns):
            raise ValueError(f"Column index {index} is out of bounds for the DataFrame.")

        # Add the indicator as a new column if indicators are not all 0 or all 1
        if not (all(x == 1 for x in missing_indicator) or all(x == 0 for x in missing_indicator)):
            new_col_name = f"{col}_missing"
            df[new_col_name] = missing_indicator

    return df



def add_power_columns(df, column_indices, power):
    """
    Adds power versions of the specified columns to the DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column_indices (list): List of column indices to be squared.
        power (int):  e.g. squaring or cubing

    Returns:
        pd.DataFrame: A new DataFrame with additional power columns.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df = df.copy()

    for index in column_indices:
        if index < 0 or index >= len(df.columns):
            raise ValueError(f"Column index {index} is out of bounds for the DataFrame.")
        
        # Handle power-specific column naming
        if power == 2:
            new_column_name = f"{column_name}_squared"
        elif power == 3:
            new_column_name = f"{column_name}_cubed"
        else:
            new_column_name = f"{column_name}_power_{power}"
        
        # Ensure the column contains numeric data
        if not pd.api.types.is_numeric_dtype(df[column_name]):
            raise ValueError(f"Column '{column_name}' is not numeric and cannot be raised to a power.")

        # Add the power-transformed column
        df[new_column_name] = df[column_name] ** power

    return df


def add_log_columns(df, column_indices):
    """
    Adds log versions of the specified columns to the DataFrame.
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column_indices (list): List of column indices to be squared.
    Returns:
        pd.DataFrame: A new DataFrame with additional log transformed columns.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df = df.copy()

    for index in column_indices:
        if index < 0 or index >= len(df.columns):
            raise ValueError(f"Column index {index} is out of bounds for the DataFrame.")
        
        if (df[index] <= 0).any():
            raise ValueError("contains non-positive values, cannot compute log.")

        column_name = df.columns[index]
        new_column_name = f"{column_name}_log"
        df[new_column_name] = np.log(df[column_name])

    return df


def add_interaction_columns(df, column_indices):
    """
    Adds interaction term of the specified columns to the DataFrame.
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column_indices (list): List of lists with two integers each.
    Returns:
        pd.DataFrame: A new DataFrame with additional interaction columns.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df = df.copy()

    for pair in column_indices:
        if len(pair) != 2:
            raise ValueError("Only consider two way interactions.")
        
        column_name = ""
        for index in pair:
            if index < 0 or index >= len(df.columns):
                raise ValueError(f"Column index {index} is out of bounds for the DataFrame.")

            column_name = column_name + "_" + df.columns[index][:5]
        new_column_name = f"{column_name}_intA"
        df[new_column_name] = df[pair[0]] * df[pair[1]]

    return df


######################################### Plan ####################################################

#built pipeline with LLM feature engeeneering and imputation and compare results
#ask Gwen which features should be used as polynomial or log transformed features
#Binning, Interaction features
#Handling Temporal Features
#Extract Date Components: Extract day, month, year, or day of the week from timestamps.
#Lag Features: Create features based on past values (useful in time-series data).
#Rolling Statistics: Compute rolling means, sums, or standard deviations over time.

#Feature Engineering plan
#Need from previous groups: (keep text strings short)
    #preprocessed data in rectangular form. Removal of unimportant, constant, duplicate features
    #addInfo = "" e.g. external knowledge
    #dataSummary = ""
    #colDescription = "" description of the columns, especially which contain time data, text data, anything unsual
    #responseVar = "mpg" 
    #task: regression or classification
    #metric e.g. MSE or misclassification rate
#Hard code: imputation, standadisation of numerical features, handling time date?,
#   log-transform features with outliers?
#Vincent: Ask Gwen about specific feature transformation, let it output column indices
#   e.g. Adding squared and cubic terms
#   what other standard feature transformation do you know?
#Tim: Ask Gwen more generally, look at the data and additional information and output feature
#   transformations as python Code, e.g. BMI as transformation of weight and height

#Output to the next group: dataframe with additional features added

#Both approaches work on a technical level
#Questions: Does this actually improve predictive power? -> testing
#   Which transformation, e.g. for time series data, can the LLM detect? -> testing






################################################ LLM queries #################################################

query_missing = "Have a look at the following columns: " + colnames_string + " . Also consider the dataframe description: " + dataSummary + " , the description of the columns: " + colDescription + ", these additional information: " + addInfo +  " and try to have an educated guess, for which variable the indicator whether the value is missing or not could have predictive power on the response variable: " + responseVar + ". Only output the column indices for which you think the column is relevant for predicting " + responseVar + ", so return a list of integers and do not output anything else! If you don't find a useful column, return NULL."
print(query_missing)
answer_missing = call_llm_mv(query_missing, "data science expert")
print(answer_missing)

query_Ints = "Have a look at the following columns: " + colnames_string + " . Also consider the dataframe description: " + dataSummary + " , the description of the columns: " + colDescription + ", these additional information: " + addInfo +  " and try to have an educated guess, for which variables an interaction term should be added as a new feature which could improve a prediction model on " + responseVar + ", so return a list of lists with two integers in each and do not output anything else! If you don't find anythign useful, return NULL."
query_Squ = "Have a look at the following columns: " + colnames_string + " . Also consider the dataframe description: " + dataSummary + " , the description of the columns: " + colDescription + ", these additional information: " + addInfo +  " and try to have an educated guess, for which variables a squared tern should be added as a new feature which could improve a prediction model on " + responseVar + ", so return a list of integers and do not output anything else!  If you don't find a useful column, return NULL."
query_Cub = "Have a look at the following columns: " + colnames_string + " . Also consider the dataframe description: " + dataSummary + " , the description of the columns: " + colDescription + ", these additional information: " + addInfo +  " and try to have an educated guess, for which variables a cubic term should be added as a new feature which could improve a prediction model on " + responseVar + ", so return a list of integers and do not output anything else!  If you don't find a useful column, return NULL."
query_Log = "Have a look at the following columns: " + colnames_string + " . Also consider the dataframe description: " + dataSummary + " , the description of the columns: " + colDescription + ", these additional information: " + addInfo +  " and try to have an educated guess,  which variable should be log-tranformed  which could improve a prediction model on " + responseVar + ", so return a list of integers and do not output anything else!  If you don't find a useful column, return NULL."

answer_Ints = call_llm_mv(query_Ints, "data science expert")
answer_Squ = call_llm_mv(query_Squ, "data science expert")
answer_Cub = call_llm_mv(query_Cub, "data science expert")
answer_Loc = call_llm_mv(query_Log, "data science expert")





############################################ TESTING ###################################################




#ampute
mtcars_df_mis = delete_values_with_exclusion(mtcars_df, 10, responseVar)
#print(mtcars_df_mis.head())

#impute
#mtcars_df_imp = impute_mixed_data(mtcars_df_mis, 1)
#print(mtcars_df_imp.head())

#complete cases
complete_cases_df = mtcars_df_mis.dropna()

#use LLM to modify the datafreame
mtcars_missCol = add_missingness_columns(mtcars_df_mis, answer_missing)
mtcars_missCol_imp = impute_mixed_data(mtcars_missCol, 3)
mtcars_missCol_imp = add_power_columns(mtcars_missCol_imp, answer_Squ, 2)
mtcars_missCol_imp = add_power_columns(mtcars_missCol_imp, answer_Cub, 3)
mtcars_missCol_imp = add_log_columns(mtcars_missCol_imp, answer_Log)
mtcars_missCol_imp = add_interaction_columns(mtcars_missCol_imp, answer_Ints)

#mtcars_inclMissingIndicator = add_missingness_correlation_vars(mtcars_df_mis, responseVar, 0.1)
#mtcars_inclMissingIndicator = add_missingness_correlation_vars(mtcars_df_mis, responseVar, 0.1)
#mtcars_inclMissingIndicator = impute_mixed_data(mtcars_inclMissingIndicator, 1)
#print(mtcars_inclMissingIndicator.head(10))

#train_and_compare(mtcars_df, mtcars_df_imp, responseVar)
#train_and_compare(mtcars_df, complete_cases_df, responseVar)
#train_and_compare(mtcars_df, mtcars_inclMissingIndicator, responseVar)


#TODO
#Test with LLM if it works technically
#Improve imputation
#write general function that works with all dataframes
