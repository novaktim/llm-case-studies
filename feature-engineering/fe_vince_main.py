import sys
sys.path.append('/feature-engineering')
from  fe_vince_functions import *
from ninept import qwen

################################################### Main function ##########################################

def vince_feature_engineering(df, 
                            eda_summary = "", #from EDA
                            ext_info = "", #from external knowledge group
                            response = "mpg"):  
    
    colnames_string = ", ".join(df.columns) 
    
    missing_frequency = df.isnull().sum().sum() / df.size
    
    #### imputation: First add missing columns for numerical variables, then impute
    query_missing = "Have a look at the following columns: " + colnames_string + " . Also consider the results from the explanatory data analysis: " + eda_summary + " , these additional information: " + ext_info +  " and try to have an educated guess, for which numerical variables the indicator whether the value is missing or not could have predictive power on the response variable: " + response + ". Ignore categorical variables. Return a list of integers and do not output anything else!  If you don't find a useful column, return NULL."
    answer_missing = call_llm_mv(query_missing, "data science expert")

    try:
        df_new = add_missingness_columns(df, answer_missing)
        print("Successfully added missingness columns.")
    except Exception as e:
        df_new = df
        print(f"Failed to add missing columns: {e}")    
        
    
    #the numfer of imputations depend on the size of the dataset and the missingness rate
    n_imputations, explanation = determine_imputations(missing_frequency, df.shape[0])
    df_new = impute_mixed_data(df_new, n_imputations = n_imputations) #this should never fail
    
    
    #### handle temporal data if the df contains temporal data
    try:
        df_new = enrich_temporal_data(df_new)
        print("Successfully handled temporal data.")
    except Exception as e:
        print(f"Failed to enrich temporal data: {e}")    
    
    
    #### ask LLM about some common transformation
    #query_Ints = "Have a look at the following columns: " + colnames_string + " . Also consider the results from the explanatory data analysis: " + eda_summary + " , these additional information: " + ext_info +  " and try to have an educated guess, for which variables an interaction term should be added as a new feature which could improve a prediction model on " + response + ", so return a list of two integers and do not output anything else! Example output: [2, 5]"
    query_Squ = "Have a look at the following columns: " + colnames_string + " . Also consider the results from the explanatory data analysis: " + eda_summary + " , these additional information: " + ext_info +  " and try to have an educated guess, for which variables a squared tern should be added as a new feature which could improve a prediction model on " + response + ", so return a small list of integers and do not output anything else!  If you don't find a useful column, return NULL. Example output: [3, 7]"
    query_Cub = "Have a look at the following columns: " + colnames_string + " . Also consider the results from the explanatory data analysis: " + eda_summary + " , these additional information: " + ext_info +  " and try to have an educated guess, for which variables a cubic term should be added as a new feature which could improve a prediction model on " + response + ", so return a small list of integers and do not output anything else!  If you don't find a useful column, return NULL. Example output: [1, 4]"
    query_Log = "Have a look at the following columns: " + colnames_string + " . Also consider the results from the explanatory data analysis: " + eda_summary + " , these additional information: " + ext_info +  " and try to have an educated guess,  which variable should be log-tranformed  which could improve a prediction model on " + response + ", so return a small list of integers and do not output anything else!  If you don't find a useful column, return NULL. Example output: [5, 9]"
   # query_boxCox = "Have a look at the following columns: " + colnames_string + " . Also consider the results from the explanatory data analysis: " + eda_summary + " , these additional information: " + ext_info +  " and try to have an educated guess,  which variable should be box-cox transformed which could improve a prediction model on " + response + ", so return a list of integers and do not output anything else!  If you don't find a useful column, return NULL."
   # query_temp = "Have a look at the following columns: " + colnames_string + " . Also consider the results from the explanatory data analysis: " + eda_summary + " , these additional information: " + ext_info +  " and tell me, which columns contain temporal data, e.g. the date, so return a list of integers and do not output anything else!  If you don't find a column with temporal data, return NULL."
    
    #answer_Ints = call_llm_mv(query_Ints, "data science expert")
    answer_Squ = call_llm_mv(query_Squ, "data science expert")
    print(answer_Squ)
    answer_Cub = call_llm_mv(query_Cub, "data science expert")
    print(answer_Cub)
    answer_Log = call_llm_mv(query_Log, "data science expert")
    #answer_temp = call_llm_mv(query_temp, "data science expert")
    #answer_boxCox = call_llm_mv(query_boxCox, "data science expert")
     
    
    #Try to perform transformation without crashing the main
    try:
        df_new = add_power_columns(df_new, answer_Squ, 2)
        print("Successfully applied squared power columns.")
    except Exception as e:
        print(f"Failed to add squared power columns: {e}")

    try:
        df_new = add_power_columns(df_new, answer_Cub, 3)
        print("Successfully applied cubed power columns.")
    except Exception as e:
        print(f"Failed to add cubed power columns: {e}")

    try:
        df_new = add_log_columns(df_new, answer_Log)
        print("Successfully applied log columns.")
    except Exception as e:
        print(f"Failed to add log columns: {e}")

    '''
    try:
        excluded_cols = []
        tmp = add_interaction_column_pair(df_new, answer_Ints)
        df_new = tmp[0]
        excluded_cols.append(tmp[1])
        print("Successfully applied interaction columns.")
    except Exception as e:
        print(f"Failed to add interaction columns: {e}")
    '''
    
    excluded_cols = []
    
    
    
    # Initial query
    query_Ints = (
        "Have a look at the following columns: " + colnames_string +
        " . Also consider the results from the explanatory data analysis: " + eda_summary +
        " , these additional information: " + ext_info +
        " and try to have an educated guess, for which 2 variables an interaction term "
        "should be added as a new feature which could improve a prediction model on " +
        response + ", so return a list of exactly two integers and do not output anything else! "
       # "Example output: [2, 5]"
    )

    max_iterations = len(colnames_string) #safety break
    while True:
        if max_iterations == 0:
            break
        max_iterations  = max_iterations - 1
        
       
        excluded_info = f" The following column pairs have already been excluded: {excluded_cols}."
        print(f"Excluded cols: {excluded_cols}")
        updated_query = query_Ints + excluded_info

        try:
            # Call the LLM to get the next pair of column indices
            answer_Ints = call_llm_mv_2(updated_query, "data science expert")

            # Validate the LLM output
            if not isinstance(answer_Ints, list) or len(answer_Ints) != 2 or not all(isinstance(i, int) for i in answer_Ints):
                #raise ValueError(f"Invalid response from LLM: {answer_Ints}. Expected a list of two integers.")
                raise ValueError(f"Invalid response from LLM:  Expected a list of two integers.")

            # Try to add the interaction column
            tmp = add_interaction_column_pair(df_new, answer_Ints)
            df_new = tmp[0]
            #excluded_cols.append(tmp[1])  # Update excluded_cols with the handled pair
            excluded_cols.extend(tmp[1] if isinstance(tmp[1], list) else [tmp[1]])  # Ensure list format
            excluded_cols = [col for pair in excluded_cols for col in pair] #flatten
            print(f"Successfully applied interaction columns for pair: {tmp[1]}")

        except ValueError as ve:
            print(f"Validation error: {ve}")
            break  # Exit if the LLM response is invalid

        except Exception as e:
            print(f"Failed to add interaction columns: {e}")
            break  # Exit on other exceptions
    
    
    #Ask LLM which transformations have been performed
    new_colnames =  ", ".join(df_new.columns)
    
    query_trafos = "Which feature engineering transformations have been done?"
    "Look at the column names" + new_colnames + "and have an educated guess "
    " based on the column name endings: "
    "the column names ending e.g. in _squared when the orginal column was added squared, _log for a "
    "log-transformation, _missing for adding a dummy encoded column indicating if the observation has "
    "a missing value in the orginal variable. _is_weekend, _day_of_week, etc, are indicators that"
    "date data was enriched with furhter information. _intA indicate an added interaction term. "
    
    answer_trafos = qwen(query_trafos)
    print(answer_trafos + "\n")

    #TODO fix interaction query x?
    #TODO research more standardish feature engineering stuff e.g. temporal data hardcoding
    #TODO save explanations
    #TODO also use these to add interactions iteratively

    
    
    
    #return transformed dataframe and a description of performed transformations
    results = {
    "transformed data": df_new,
    "explanation": answer_trafos + explanation
    }
    return results


#### short test
file_path = 'data/mtcars.csv'  
mtcars_df = pd.read_csv(file_path)

#mtcars_df = mtcars_df.drop(mtcars_df.columns[0], axis=1)
mtcars_df_mis = delete_values_with_exclusion(mtcars_df, 25, "mpg")
print(mtcars_df_mis)

results = vince_feature_engineering(mtcars_df_mis)
#print(results["transformed data"])
results["transformed data"].to_csv("test_new.csv", index=False)