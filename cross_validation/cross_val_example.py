import numpy as np
from helper.classifications_cv import hold_out_split_and_train
from sklearn.linear_model import LinearRegression


model = LinearRegression()

result = hold_out_split_and_train("Datasets/Timeseries Dataset/Microsoft_Stock.csv",{"test_size":0.2},model,"regression")

print(result)