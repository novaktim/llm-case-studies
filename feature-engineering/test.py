
import sys
sys.path.append('/feature-engineering')
from  fe_vince_functions import *
from  FE_main import *
from ninept import qwen

#### short test
# file_path = 'data/mtcars.csv'  
# mtcars_df = pd.read_csv(file_path)

# #mtcars_df = mtcars_df.drop(mtcars_df.columns[0], axis=1)
# mtcars_df_mis = delete_values_with_exclusion(mtcars_df, 25, "mpg")
# print(mtcars_df_mis)

# results = fe_main(mtcars_df_mis, response = "mpg", eda_summary = "", ext_info = "")
# print("########################### results ################################### \n")
# print(results["fe_summary"])
# results["df_new"].to_csv("mtcars_test_final.csv", index=False)


# #### short test 2
file_path = 'data/boston_housing.csv'  
bost_df = pd.read_csv(file_path)
bost_df_mis = delete_values_with_exclusion(bost_df, 25, "crim")
print(bost_df_mis)

results = fe_main(bost_df_mis, response = "crim", eda_summary = "", ext_info = "")
print("########################### results ################################### \n")
print(results["fe_summary"])
results["df_new"].to_csv("bost_test_final.csv", index=False)


#### titanic test
ext = ("The Spaceship Titanic competition on Kaggle is a machine learning challenge that invites " +
"participants to predict which passengers were transported to an alternate dimension during the" +
"spaceship's collision with a spacetime anomaly. This scenario is inspired by the famous Titanic " +
"disaster, reimagined in a futuristic space setting." +
"Participants are tasked with developing a model that can accurately determine whether a "
"passenger was transported to another dimension based on the available data. " +
"The challenge serves as an excellent opportunity for both beginners and experienced data " +
"scientists to apply and enhance their predictive modeling skills.")


file_path = 'data/train.csv'  
titanic_df = pd.read_csv(file_path)
results = fe_main(titanic_df, response = "Transported", eda_summary = "", ext_info = ext)
print("########################### results ################################### \n")
print(results["fe_summary"])
results["df_new"].to_csv("tit_test_final.csv", index=False)


#TODO issues
# cant use response variable for interactions
# remove break statements? stops too early when tries for cat. variable