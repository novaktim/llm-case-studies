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
    
    
    #### imputation: First add missing columns for numerical variables, then impute
    query_missing = "Have a look at the following columns: " + colnames_string + " . Also consider the results from the explanatory data analysis: " + eda_summary + " , these additional information: " + ext_info +  " and try to have an educated guess, for which numerical variables the indicator whether the value is missing or not could have predictive power on the response variable: " + response + ". Ignore categorical variables. Return a list of integers and do not output anything else!  If you don't find a useful column, return NULL."
    answer_missing = call_llm_mv(query_missing, "data science expert")

    try:
        df_new = add_missingness_columns(mtcars_df_mis, answer_missing)
        print("Successfully added missingness columns.")
    except Exception as e:
        print(f"Failed to add missing columns: {e}")    
        
    df_new = impute_mixed_data(df, n_imputation = 5) #this should never fail
    
    
    #### handle temporal data if the df contains temporal data
    try:
        df_new = enrich_temporal_data(df_new)
        print("Successfully enriched temporal data.")
    except Exception as e:
        print(f"Failed to enrich temporal data: {e}")    
    
    
    #### ask LLM about some common transformation
    query_Ints = "Have a look at the following columns: " + colnames_string + " . Also consider the results from the explanatory data analysis: " + eda_summary + " , these additional information: " + ext_info +  " and try to have an educated guess, for which variables an interaction term should be added as a new feature which could improve a prediction model on " + response + ", so return a list of lists with two integers in each and do not output anything else! Example output: [[2, 3], [2,4], [1,6]]"
    query_Squ = "Have a look at the following columns: " + colnames_string + " . Also consider the results from the explanatory data analysis: " + eda_summary + " , these additional information: " + ext_info +  " and try to have an educated guess, for which variables a squared tern should be added as a new feature which could improve a prediction model on " + response + ", so return a list of integers and do not output anything else!  If you don't find a useful column, return NULL."
    query_Cub = "Have a look at the following columns: " + colnames_string + " . Also consider the results from the explanatory data analysis: " + eda_summary + " , these additional information: " + ext_info +  " and try to have an educated guess, for which variables a cubic term should be added as a new feature which could improve a prediction model on " + response + ", so return a list of integers and do not output anything else!  If you don't find a useful column, return NULL."
    query_Log = "Have a look at the following columns: " + colnames_string + " . Also consider the results from the explanatory data analysis: " + eda_summary + " , these additional information: " + ext_info +  " and try to have an educated guess,  which variable should be log-tranformed  which could improve a prediction model on " + response + ", so return a list of integers and do not output anything else!  If you don't find a useful column, return NULL."
   # query_boxCox = "Have a look at the following columns: " + colnames_string + " . Also consider the results from the explanatory data analysis: " + eda_summary + " , these additional information: " + ext_info +  " and try to have an educated guess,  which variable should be box-cox transformed which could improve a prediction model on " + response + ", so return a list of integers and do not output anything else!  If you don't find a useful column, return NULL."
   # query_temp = "Have a look at the following columns: " + colnames_string + " . Also consider the results from the explanatory data analysis: " + eda_summary + " , these additional information: " + ext_info +  " and tell me, which columns contain temporal data, e.g. the date, so return a list of integers and do not output anything else!  If you don't find a column with temporal data, return NULL."
    
    answer_Ints = call_llm_pairlist(query_Ints, "data science expert")
    answer_Squ = call_llm_mv(query_Squ, "data science expert")
    answer_Cub = call_llm_mv(query_Cub, "data science expert")
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

    try:
        df_new = add_interaction_columns(df_new, answer_Ints)
        print("Successfully applied interaction columns.")
    except Exception as e:
        print(f"Failed to add interaction columns: {e}")


    #Ask LLM which transfornations have been performed
    new_colnames =  ", ".join(df_new.columns)
    query_trafos = "We have performed a few feature engineering transformations, as indicated by the column names ending e.g. in _Squ when the orginal column was added squared, _log for a logtransformation etc. Compare the orginal column names: " + colnames_string + " with the new column names: " + new_colnames + "and describe the performed transformation very briefly!"
    answer_trafos = qwen(query_trafos)
    print(answer_trafos + "\n")

    #TODO fix interaction query
    #TODO research more standardish feature engineering stuff e.g. temporal data hardcoding
    
    
    #return transformed dataframe and a description of performed transformations
    return list(df_new, answer_trafos)