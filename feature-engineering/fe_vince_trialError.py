import sys
sys.path.append('/feature-engineering')
import fe_vince_functions


############################################ Data ##########################################


# Additional information comes as imput from other group
addInfo = ""
dataSummary = ""
colDescription = ""
#description as dictionary?
responseVar = "mpg" #testing data
# Load the CSV file
file_path = 'data/mtcars.csv'  
mtcars_df = pd.read_csv(file_path)

mtcars_df = mtcars_df.drop(mtcars_df.columns[0], axis=1)
colnames_string = ", ".join(mtcars_df.columns)
# original data
#print(mtcars_df.head())



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
#print(query_missing)
answer_missing = call_llm_mv(query_missing, "data science expert")
#print(answer_missing)

query_Ints = "Have a look at the following columns: " + colnames_string + " . Also consider the dataframe description: " + dataSummary + " , the description of the columns: " + colDescription + ", these additional information: " + addInfo +  " and try to have an educated guess, for which variables an interaction term should be added as a new feature which could improve a prediction model on " + responseVar + ", so return a list of lists with two integers in each and do not output anything else! Example output: [[2, 3], [2,4], [1,6]]"
query_Squ = "Have a look at the following columns: " + colnames_string + " . Also consider the dataframe description: " + dataSummary + " , the description of the columns: " + colDescription + ", these additional information: " + addInfo +  " and try to have an educated guess, for which variables a squared tern should be added as a new feature which could improve a prediction model on " + responseVar + ", so return a list of integers and do not output anything else!  If you don't find a useful column, return NULL."
query_Cub = "Have a look at the following columns: " + colnames_string + " . Also consider the dataframe description: " + dataSummary + " , the description of the columns: " + colDescription + ", these additional information: " + addInfo +  " and try to have an educated guess, for which variables a cubic term should be added as a new feature which could improve a prediction model on " + responseVar + ", so return a list of integers and do not output anything else!  If you don't find a useful column, return NULL."
query_Log = "Have a look at the following columns: " + colnames_string + " . Also consider the dataframe description: " + dataSummary + " , the description of the columns: " + colDescription + ", these additional information: " + addInfo +  " and try to have an educated guess,  which variable should be log-tranformed  which could improve a prediction model on " + responseVar + ", so return a list of integers and do not output anything else!  If you don't find a useful column, return NULL."
query_boxCox = "Have a look at the following columns: " + colnames_string + " . Also consider the dataframe description: " + dataSummary + " , the description of the columns: " + colDescription + ", these additional information: " + addInfo +  " and try to have an educated guess,  which variable should be box-cox transformed which could improve a prediction model on " + responseVar + ", so return a list of integers and do not output anything else!  If you don't find a useful column, return NULL."

print(query_Ints)
answer_Ints = call_llm_pairlist(query_Ints, "data science expert")
print(answer_Ints)

answer_Squ = call_llm_mv(query_Squ, "data science expert")
answer_Cub = call_llm_mv(query_Cub, "data science expert")
answer_Log = call_llm_mv(query_Log, "data science expert")
answer_boxCox = call_llm_mv(query_Log, "data science expert")





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
#Improve imputation x
    #research bayesian ridge 
#write general function that works with all dataframes
#LLM pairlist check does not work - just do 2 at a time? keeping track?
#Add description of feature engineering as output
#Test with real EDA results


