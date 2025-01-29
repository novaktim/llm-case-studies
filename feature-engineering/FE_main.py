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
    
    print("Performing imputation and hard coded standard feature engineering steps.\n")
    fe_vince_results = vince_feature_engineering(df, eda_summary, ext_info, response) #including imputation
    df_new = fe_vince_results[0]
    trafos_summary = fe_vince_results[1]
    
    print("Performing flexible feature engineering steps.\n")
    df_new, generation_info = feature_generation.feature_generation(df_new, eda_summary, ext_info, response)
    
    #append trafos_summary from vince and tim
    
    return list(df_new, trafos_summary, generation_info)



#TODO
#implementation summary return for model selection
#description, what feature engineering what was done
#description what we did with missing values
#test with real EDA results - tell them to push

    
