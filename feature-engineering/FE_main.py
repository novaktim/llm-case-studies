#### This is the function supposed to be called in the overall pipeline ######

import sys
import pandas
sys.path.append('/feature-engineering')
from  fe_vince_main import *
import feature_generation #tim
#1 call imputation
#2 call vince FE approach
#3 call Tim FE approach
#4 aggregate and return

##### info for the preprocessing group:
# if the data contains temporal data, e.g. the date, it should be in datetime64_any_dtype format.
# See enrich_temporal_data() function in fe_vince_functions.py for more details

#fe_main performs feature engineering
def fe_main(df, eda_summary, ext_info, response): 
    """
    Main function to apply feature engineering to a dataset.

    This function performs both standard (hardcoded) and flexible feature engineering steps,
    including data imputation and feature generation. It integrates multiple components of
    the feature engineering pipeline to produce a transformed dataset ready for modeling.

    Parameters:
        df (pd.DataFrame): 
            The input dataset to be processed. This is typically raw data requiring cleaning 
            and transformation.
        eda_summary (str): 
            A summary of exploratory data analysis (EDA) results. Includes insights such as 
            missing value statistics or other metadata derived from EDA.
        ext_info (str): 
            External information that supplements the dataset, such as domain knowledge or 
            additional contextual details.
        response (str): 
            The name of the target variable in the dataset. Used during transformations to ensure
            relevance for predictive modeling tasks.

    Returns:
        list: A list containing three elements:
              1. df_new (pd.DataFrame): The transformed dataset after applying all feature engineering steps.
              2. trafos_summary (str): A summary of transformations applied during standard
                 feature engineering steps (e.g., imputations).
              3. generation_summary (str): Metadata about newly generated features during flexible
                 feature engineering steps.
    """
    
    print("Performing imputation and hard coded standard feature engineering steps.\n")
    fe_vince_results = vince_feature_engineering(df, eda_summary, ext_info, response) #including imputation
    df_new = fe_vince_results[0]
    trafos_summary = fe_vince_results[1]
    
    print("Performing flexible feature engineering steps.\n")
    df_new, generation_summary = feature_generation.feature_generation(df_new, eda_summary, ext_info, response)
    
    return list(df_new, trafos_summary, generation_summary)



#TODO
#implementation summary return for model selection
#description, what feature engineering what was done
#description what we did with missing values
#test with real EDA results - tell them to push

    
