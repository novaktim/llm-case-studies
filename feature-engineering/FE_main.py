#### This is the function supposed to be called in the overall pipeline ######

import sys
import pandas
sys.path.append('/feature-engineering')
import fe_vince_main
import feature_generation
#import feature_generation #tim
#1 call imputation
#2 call vince FE approach
#3 call Tim FE approach
#4 aggregate and return

#fe_main performs feature engineering
def fe_main(df, eda_summary, ext_info, response): 
    
    print("Performing imputation and hard coded standard feature engineering steps")
    df_new = vince_feature_engineering(df, eda_summary, ext_info, response) #including imputation
    df_new = tim_feature_engineering(df_new, eda_summary, ext_info, response)
    
    
    return df_new



# from ..data.medical_dataset import medical_datas
# df = medical_dataset()

df = pandas.read_csv("EDA/EDA/melb_data.csv")
transformedDataset = feature_generation.feature_generation(df)

print("Finished")

#TODO
#implementation summary return for model selection
#description, what feature engineering what was done
#description what we did with missing values
#test with real EDA results - tell them to push

    
