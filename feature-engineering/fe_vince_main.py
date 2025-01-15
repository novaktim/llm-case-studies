import sys
sys.path.append('/feature-engineering')
import fe_vince_functions


################################################### Main function ##########################################

def vince_feature_engineering(df, 
                            eda_summary = "", #from EDA
                            ext_info = "", #from external knowledge group
                            response = "mpg"):  
    
    colnames_string = ", ".join(df.columns) #might delete this if we get this from EDA
    
    #### imputation
    query_missing = "Have a look at the following columns: " + colnames_string + " . Also consider the results from the explanatory data analysis: " + eda_summary + " , these additional information: " + ext_info +  " and try to have an educated guess, for which variable the indicator whether the value is missing or not could have predictive power on the response variable: " + response + "so return a list of integers and do not output anything else!  If you don't find a useful column, return NULL."
    answer_missing = call_llm_mv(query_missing, "data science expert")
    df_new = add_missingness_columns(mtcars_df_mis, answer_missing)
    df_new = impute_mixed_data(df, n_imputation = 5) 
    #need to first impute categorical, than ask LLM, then impute numerical
    
    
    query_Ints = "Have a look at the following columns: " + colnames_string + " . Also consider the results from the explanatory data analysis: " + eda_summary + " , these additional information: " + ext_info +  " and try to have an educated guess, for which variables an interaction term should be added as a new feature which could improve a prediction model on " + response + ", so return a list of lists with two integers in each and do not output anything else! Example output: [[2, 3], [2,4], [1,6]]"
    query_Squ = "Have a look at the following columns: " + colnames_string + " . Also consider the results from the explanatory data analysis: " + eda_summary + " , these additional information: " + ext_info +  " and try to have an educated guess, for which variables a squared tern should be added as a new feature which could improve a prediction model on " + response + ", so return a list of integers and do not output anything else!  If you don't find a useful column, return NULL."
    query_Cub = "Have a look at the following columns: " + colnames_string + " . Also consider the results from the explanatory data analysis: " + eda_summary + " , these additional information: " + ext_info +  " and try to have an educated guess, for which variables a cubic term should be added as a new feature which could improve a prediction model on " + response + ", so return a list of integers and do not output anything else!  If you don't find a useful column, return NULL."
    query_Log = "Have a look at the following columns: " + colnames_string + " . Also consider the results from the explanatory data analysis: " + eda_summary + " , these additional information: " + ext_info +  " and try to have an educated guess,  which variable should be log-tranformed  which could improve a prediction model on " + response + ", so return a list of integers and do not output anything else!  If you don't find a useful column, return NULL."
   # query_boxCox = "Have a look at the following columns: " + colnames_string + " . Also consider the results from the explanatory data analysis: " + eda_summary + " , these additional information: " + ext_info +  " and try to have an educated guess,  which variable should be box-cox transformed which could improve a prediction model on " + response + ", so return a list of integers and do not output anything else!  If you don't find a useful column, return NULL."
    query_temp = "Have a look at the following columns: " + colnames_string + " . Also consider the results from the explanatory data analysis: " + eda_summary + " , these additional information: " + ext_info +  " and tell me, which columns contain temporal data, e.g. the date, so return a list of integers and do not output anything else!  If you don't find a column with temporal data, return NULL."
    
    answer_Ints = call_llm_pairlist(query_Ints, "data science expert")
    answer_Squ = call_llm_mv(query_Squ, "data science expert")
    answer_Cub = call_llm_mv(query_Cub, "data science expert")
    answer_Log = call_llm_mv(query_Log, "data science expert")
    answer_missing = call_llm_mv(query_missing, "data science expert")
    answer_temp = call_llm_mv(query_temp, "data science expert")
    #answer_boxCox = call_llm_mv(query_boxCox, "data science expert")
     
    
    #TODO add try catch stuff here to be more robust
    df_new = add_missingness_columns(mtcars_df_mis, answer_missing)
    df_new = add_power_columns(mtcars_missCol_imp, answer_Squ, 2)
    df_new = add_power_columns(mtcars_missCol_imp, answer_Cub, 3)
    df_new = add_log_columns(mtcars_missCol_imp, answer_Log)
    df_new = add_interaction_columns(mtcars_missCol_imp, answer_Ints)
    #df_new = add_boxcox_columns(mtcars_missCol_imp, answer_boxCox)

    #TODO need to return information about the performed transformations as well
    #need character string preferebly
    is_not_null = {
    "Int_": answer_Ints is not None,
    "answer_Squ": answer_Squ is not None,
    "answer_Cub": answer_Cub
}
    


    #TODO fix interaction query
    #TODO reasearch more standardish feature engineering stuff e.g. temporal data hardcoding
    return df_new