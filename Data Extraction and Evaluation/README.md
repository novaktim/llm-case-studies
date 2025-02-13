# Data Extraction and Evaluation:
## Introduction and Setup:
The data extraction procedure is done using Kaggle API to retrieve the datasets from the active competitions from Kaggle Competition. After getting the dataset from Kaggle Competition, we retreive the metadata of the Kaggle Competition which contains all the information regarding that particular competition and the evaluation criteria for that competition. Finally, all of this metadata(information) is given to our Qwen LLM which will generate us the code for the evaluation criteria for the competition and our code will create a new llm_evaluation_code.py file on the root location which is basically our final output.
## Requirements:
You need to first create an account on Kaggle and you need to get the Kaggle API in order to retrieve the datasets from Kaggle Competition. You can use the following steps to get Kaggle API. After creating an account, go to settings, then go to API and create a new token and download it.
Then you need to accept the terms and conditions for the Kaggle Competitions otherwise you cannot download the datasets using Kaggle API. We have narrowed down to six competitions and these are the links:
        https://www.kaggle.com/competitions/digit-recognizer <br />
        https://www.kaggle.com/competitions/equity-post-HCT-survival-predictions <br />
        https://www.kaggle.com/competitions/home-data-for-ml-course <br />
        https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques <br />
        https://www.kaggle.com/competitions/spaceship-titanic <br />
        https://www.kaggle.com/competitions/store-sales-time-series-forecasting <br />
Now, make a folder named .config in the root folder, then create a kaggle folder inside .config folder, and then paste your Kaggle API over there. <br />
Now, simply run DEE_Main.py and it will generate the final output.

## Output:
Firstly, the DEE_Main.py script will download all six competions in the root folder inside a new kaggle_dataset folder. Then it will generate kaggle_competition_details.txt. <br />
Afterward, it will ask you to choose one of the competitions and you can simply name the competition which you want to be evaluated. Then, it will generate the evaluation criteria output as a new llm_evaluation_code.py file.

I have pasted an output from the evaluation criteria for one of the competitions below: <br />

Available datasets:
- digit-recognizer
- equity-post-HCT-survival-predictions
- home-data-for-ml-course
- house-prices-advanced-regression-techniques
- spaceship-titanic
- store-sales-time-series-forecasting

Enter the dataset name: digit-recognizer
Hello! This is your QWEN LLM Agent !!!
Competition Details Title: Digit Recognizer Description: Learn computer vision fundamentals with the famous MNIST data URL: https://www.kaggle.com/competitions/digit-recognizer Reward: Knowledge Category: Getting Started Tags: [tabular, image, multiclass classification, categorizationaccuracy] Evaluation Metric: Categorization Accuracy. Give me a python code for implementing the evaluation metric?
Sure! To implement the evaluation metric for the Digit Recognizer competition on Kaggle, you'll need to use the provided test dataset and then calculate the categorization accuracy by comparing your predicted labels with the actual labels. Here's a step-by-step explanation and Python code using scikit-learn.

First, make sure you have installed the required libraries:
```bash
pip install numpy pandas scikit-learn
```
Now, let's write the Python code for loading the data, making predictions, and calculating the categorization accuracy:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Prepare the data
X_train = train_data.drop('label', axis=1).values.astype('float32') / 255
y_train = train_data['label'].values
X_test = test_data.values.astype('float32') / 255

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Train a simple Logistic Regression model
model = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)

# Make predictions on the validation set and the test set
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

# Calculate validation set accuracy
val_accuracy = accuracy_score(y_val, y_val_pred)
print("Validation Set Accuracy:", val_accuracy)

# Save the predicted labels for submission
submission = pd.DataFrame({'ImageId': np.arange(1, len(y_test_pred) + 1), 'Label': y_test_pred})
submission.to_csv('submission.csv', index=False)
```

This code uses a simple Logistic Regression model for demonstration purposes. In practice, you might want to try more advanced models like Convolutional Neural Networks (CNNs) for better performance. The evaluation metric is the categorization accuracy, which is calculated using the `accuracy_score` function from scikit-learn.

Remember to replace `'train.csv'` and `'test.csv'` with the paths to the actual files downloaded from the Kaggle competition page.
