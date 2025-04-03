import sys
sys.path.append('/feature-engineering')
from  fe_standard_functions import *
from ninept import qwen
import math
################################################### Main function ##########################################

def standard_feature_engineering(df, 
                            eda_summary = "", #from EDA
                            ext_info = "", #from external knowledge group
                            response = "mpg",
                            print_details = False):  #turn this on for debugging purposes
    
    
    def filter_numerical_and_response(df, response):
        """
        Filters the DataFrame to retain only numerical columns and the specified response column.

        Parameters:
            df (pd.DataFrame): The input DataFrame.
            response (str): The name of the response column to keep.

        Returns:
            pd.DataFrame: Filtered DataFrame with only numerical columns and the response column.
        """
        numeric_cols = df.select_dtypes(include=["number"]).columns
        selected_cols = list(numeric_cols) + [response] if response in df.columns else list(numeric_cols)
    
        return df[selected_cols]

    #### handle temporal data if the df contains temporal data
    try:
        df = enrich_temporal_data(df)
        print("Successfully handled temporal data.")
    except Exception as e:
        print(f"Failed to enrich temporal data: {e}")  


    #df_new = df.copy()
    
    df_new = filter_numerical_and_response(df, response=response)
    colnames_string = ", ".join(df_new.columns) 
    col_num = len(df_new.columns)    
    
  
    
    
    #### ask LLM about some common transformation
    #query_Ints = "Have a look at the following columns: " + colnames_string + " . Also consider the results from the explanatory data analysis: " + eda_summary + " , these additional information: " + ext_info +  " and try to have an educated guess, for which variables an interaction term should be added as a new feature which could improve a prediction model on " + response + ", so return a list of two integers and do not output anything else! Example output: [2, 5]"
    query_Squ = ("Have a look at the following columns: " + colnames_string + 
                 " . Also consider the results from the explanatory data analysis: " + eda_summary + 
                 " , these additional information: " + ext_info +  
                 " and try to have an educated guess, for which variables a squared tern should be added as a new feature which could improve a prediction model on " + response +
                 ", so return a small list of integers and do not output anything else!  If you don't find a useful column, return NULL. Example output: [3, 7]")
    
    query_Cub = ("Have a look at the following columns: " + colnames_string + 
                 " . Also consider the results from the explanatory data analysis: " + eda_summary + 
                 " , these additional information: " + ext_info +  
                 " and try to have an educated guess, for which variables a cubic term should be added as a new feature which could improve a prediction model on " + response +
                 ", so return a small list of integers and do not output anything else!  If you don't find a useful column, return NULL. Example output: [1, 4]")
    
    query_Log = ("Have a look at the following columns: " + colnames_string + 
                 " . Also consider the results from the explanatory data analysis: " + eda_summary + 
                 " , these additional information: " + ext_info +  
                 " and try to have an educated guess,  which variable should be log-tranformed  which could improve a prediction model on " + response + 
                 ", so return a small list of integers and do not output anything else!  If you don't find a useful column, return NULL. Example output: [5, 9]")
    
     
    
    #Try to perform transformation without crashing the main
    try:
        answer_Squ = call_llm_mv(query_Squ, "data science expert")
        df_new = add_power_columns(df_new, answer_Squ, 2, response=response)
        print("Successfully applied squared power columns.")
    except Exception as e:
        print(f"Failed to add squared power columns: {e}, will try again!")
        query_Squ = "Try again!" + query_Squ
        try:
            answer_Squ = call_llm_mv(query_Squ, "data science expert")
            df_new = add_power_columns(df_new, answer_Squ, 2, response=response)
        except:
            print("did not work again")

    try:
        answer_Cub = call_llm_mv(query_Cub, "data science expert")
        df_new = add_power_columns(df_new, answer_Cub, 3, response=response)
        print("Successfully applied cubed power columns.")
    except Exception as e:
        print(f"Failed to add cubed power columns: {e}")

    try:
        answer_Log = call_llm_mv(query_Log, "data science expert")
        df_new = add_log_columns(df_new, answer_Log, response=response)
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
    
    
    
    # Initial query
    query_Ints = (
        "Have a look at the following columns: " + colnames_string +
        " . Also consider the results from the explanatory data analysis: " + eda_summary +
        " , these additional information: " + ext_info +
        " and try to have an educated guess, for which 2 variables an interaction term "
        "should be added as a new feature which could improve a prediction model on " +
        response + ", so return a list of exactly two integers and do not output anything else! Example output: [2, 5]")
       # "Example output: [2, 5]"
    
    excluded_cols = []
    max_iterations = col_num #safety break
    while True:
        if max_iterations == 0:
            break
        max_iterations  = max_iterations - 1
        if len(excluded_cols) > 2 * math.ceil(math.sqrt(col_num)):
            #print("Enough interactions added.")
            break
            
        excluded_info = f" The following column pairs have already been excluded: {excluded_cols}."
        
        if print_details:
            print(f"Excluded cols: {excluded_cols}")
        updated_query = query_Ints + excluded_info

        try:
            # Call the LLM to get the next pair of column indices
            answer_Ints = call_llm_mv_2(updated_query, "data science expert")
            if print_details:
                print(f"Number of iterations left: {max_iterations}")
                print(answer_Ints)
            # Validate the LLM output
            if not isinstance(answer_Ints, list) or len(answer_Ints) != 2 or not all(isinstance(i, int) for i in answer_Ints):
                raise ValueError(f"Invalid response from LLM:  Expected a list of two integers.")

            # Try to add the interaction column
            try:
                tmp = add_interaction_column_pair(df_new, answer_Ints, response=response)
                df_new = tmp[0]
                new_excluded_cols = tmp[1]

                # extend
                excluded_cols.extend(new_excluded_cols if isinstance(new_excluded_cols, list) else [new_excluded_cols])
                excluded_cols = [col for pair in excluded_cols for col in (pair if isinstance(pair, (list, tuple)) else [pair])] #flattten
                
                print(f"Successfully applied interaction columns for pair: {new_excluded_cols}")
            except ValueError as e:
                print(f"Could not add Interaction column: {e}")
                
        except ValueError as ve:
            print(f"Validation error: {ve}")
            #break  # Exit if the LLM response is invalid

        except Exception as e:
            print(f"Failed to add interaction columns: {e}")
            #break  # Exit on other exceptions
    
    #delete duplicate columns, this can sometimes happen unfortunately
    df_new = df_new.loc[:, ~df_new.T.duplicated()]
    
    #Ask LLM which transformations have been performed
    new_colnames =  ", ".join(df_new.columns)
    
    query_trafos = ("We have performed a few feature engineering transformations, as indicated by the column names ending e.g. " +
                    "in _Squ when the orginal column was added squared, _log for a logtransformation etc. Compare the orginal column names: " + colnames_string + 
                    " with the new column names: " + new_colnames + "and describe the performed transformation very briefly!")
    answer_trafos = qwen(query_trafos)
    
    if print_details:
        print(answer_trafos + "\n")
    
    # merge with original
    merged_df = pd.concat([df, df_new], axis=1)
    merged_df = merged_df.loc[:, ~merged_df.T.duplicated()]
    
    #return transformed dataframe and a description of performed transformations
    results = {
    "transformed data": merged_df,
    "explanation": answer_trafos
    }
    return results


def imputation_by_LLM(df, 
                            eda_summary = "", #from EDA
                            ext_info = "", #from external knowledge group
                            response = "mpg",
                            print_details = False):
    df_new = df.copy()
    
    colnames_string = ", ".join(df_new.columns) 
    
        #### imputation: First add missing columns for numerical variables, then impute
    query_missing = ("Have a look at the following columns: " + colnames_string + 
                     " . Also consider the results from the explanatory data analysis: " + eda_summary +
                     " , these additional information: " + ext_info +  
                     " and try to have an educated guess, for which numerical variables the indicator whether the value is missing or not could have predictive power on the response variable: " + response + 
                     ". Ignore categorical variables. Return a list of integers and do not output anything else!  If you don't find a useful column, return NULL.")
    try:
        answer_missing = call_llm_mv(query_missing, "data science expert")
    except Exception:
        print("Could not ask LLM for missingness columns.")
        answer_missing = ""
    if print_details:
        print(answer_missing)
        
    try:
        df_new = add_missingness_columns(df_new, answer_missing, response=response)
        print("Successfully added missingness columns.")
    except Exception as e:
        print(f"Failed to add missing columns: {e}")    
        
    
    #the number of imputations depend on the size of the dataset and the missingness rate
    missing_frequency = df.isnull().sum().sum() / df.size
    #n_imputations, explanation = determine_imputations(missing_frequency, df.shape[0])
    n_imputations = 1 #more robust
    
    try:
        df_new = impute_mixed_data(df_new, n_imputations = n_imputations) #this should never fail
    except:
        print("Imputations failed")
    return df_new

#TODO
# make own reporting string
# write about dummy encoding
# add to query that it should not change the number of rows