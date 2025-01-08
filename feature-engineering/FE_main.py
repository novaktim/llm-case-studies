#### This is the function supposed to be called in the overall pipeline ######

import sys
sys.path.append('/feature-engineering')
import vince
import feature_generation
#1 call imputation
#2 call vince FE approach
#3 call Tim FE approach
#4 aggregate and return


def fe_main(df, eda_summary, ext_info, response): 
    df_new = vince_feature_engineering(df, eda_summary, ext_info, response)
    df_new = tim_feature_engineering(df_new, eda_summary, ext_info, response)
    
    return df_new

#TODO
#implementation summary return for model selection
#description, what feature engineering what was done
#description what we did with missing values
    

    
    