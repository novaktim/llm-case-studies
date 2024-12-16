# llm-case-studies
## EDA
### Performing EDA steps:
To get a response from the Gwen model, first, we performed the following steps:
- Dataset information (Shape and type)
- Missing value analysis
- Drop high uniqueness categorical column (Threshold = 0.7)
- Find constant columns and duplicate rows
- Distribution analysis of Numerical features (Skewness and Kurtosis)
- Boxplot Summary Information (Min, Q1, Median, Q3, Max) and outliers based on IQR
- Correlation analysis 
- Categorical value analysis
### Using the Qwen model
Merging all the information into a single field, use this field as **Content** of the Qwen Model.
### Retrieving the information from the Qwen Model
From the Qwen model, get the response from the given content.
### Currently working on:
- Encoding the categorical variable
### Author
Ahmed Arian Sajid
