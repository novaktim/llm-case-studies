
import sys
sys.path.append('/feature-engineering')
from  fe_vince_functions import *
from  FE_main import *
from ninept import qwen

#### short test
file_path = 'data/mtcars.csv'  
mtcars_df = pd.read_csv(file_path)

#mtcars_df = mtcars_df.drop(mtcars_df.columns[0], axis=1)
mtcars_df_mis = delete_values_with_exclusion(mtcars_df, 25, "mpg")
print(mtcars_df_mis)

results = fe_main(mtcars_df_mis, response = "mpg", eda_summary = "", ext_info = "")
print("########################### results ################################### \n")
print(results["fe_summary"])
results["df_new"].to_csv("mtcars_test_final.csv", index=False)


#### short test 2
file_path = 'data/boston_housing.csv'  
bost_df = pd.read_csv(file_path)

#mtcars_df = mtcars_df.drop(mtcars_df.columns[0], axis=1)
bost_df_mis = delete_values_with_exclusion(bost_df, 25, "crim")
print(bost_df_mis)

results = fe_main(bost_df_mis, response = "crim", eda_summary = "", ext_info = "")
print("########################### results ################################### \n")
print(results["fe_summary"])
results["df_new"].to_csv("bost_test_final.csv", index=False)

