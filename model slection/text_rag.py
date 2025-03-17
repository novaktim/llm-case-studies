from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
import json
#from ninept import qwen

os.environ["TOKENIZERS_PARALLELISM"] = "false"
file_path = "selection_guide.pdf"  
loader = PyPDFLoader(file_path)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_documents = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings()

docsearch = FAISS.from_documents(split_documents, embeddings)

#docsearch.save_local("faiss_index")

#docsearch = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

groq_api_key = "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"

llm = ChatGroq(
    temperature=0,
    groq_api_key=groq_api_key,
    model_name="deepseek-r1-distill-llama-70b"
)


qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=docsearch.as_retriever(), return_source_documents=True)

problem_prompt = '''
Determine the price of a car, given the features like model, year, km, etc.
'''

data_prompt = '''
The dataset contains 2059 observations and 20 variables with no duplicate rows. The data types include integers, floats, and objects (strings). There are no constant columns, indicating no variables with identical values across all rows.

Data quality aspects:

Missing Values: Some columns have missing values. Engine, Max Power, Max Torque, Drivetrain, Length, Width, Height, Seating Capacity, and Fuel Tank Capacity have missing data ranging from 3.1% to 6.6%. This may impact the analysis, especially if these variables are important for your study. You can handle missing values by imputation (e.g., mean, median, or using a model) or exclusion of the corresponding rows or columns, depending on the context and significance of the missing data.
High Uniqueness Columns: No high-unicity columns were detected, which means there aren't many columns with a large number of unique values relative to the number of rows. This suggests that the data is not overly sparse.
Outliers: The numeric features Price, Year, Kilometer, Length, Width, Height, Seating Capacity, and Fuel Tank Capacity exhibit outliers, as indicated by boxplot summaries. Outliers can skew statistical analyses, so they should be investigated and potentially addressed (e.g., by removing extreme values, winsorizing, or using robust methods).
Correlation: Some variables show moderate to high correlation, such as Price with Length and Width, and Fuel Tank Capacity with Length and Width. This might indicate multicollinearity, which can affect model performance. Consider using techniques like PCA or variance inflation factors (VIF) to address this issue.
Categorical Variables: Several categorical columns have a large number of unique values, like Model and Engine, which could lead to "the curse of dimensionality" if used directly in models. Consider one-hot encoding or feature engineering techniques to reduce the dimensionality.
In summary, while the dataset has some missing values and outliers, it seems generally suitable for analysis after addressing these issues through appropriate data preprocessing and handling of missing values. Additionally, be cautious about multicollinearity and high-dimensional categorical variables when building models.
'''

system_prompt = '''
With the help of the document provided, give the best model(s) for the use case of above mentioned problem and data.
write the name of the model, a python function that takes train and test datasets as input for implementation 
of that model and outputs the prediction of that model by training the model with train_data input, and a short summarization of each model, aka. why it's been selected. 
the final output will only be a JSON file which has model, function and explanation as keys with two or more suitable models.
'''

result = qa_chain({"query": problem_prompt+data_prompt+system_prompt})

print("\n")
print("Answer:", result["result"])
print("JSON:", json.loads(result["result"].split("```json\n")[1].split("```")[0]))




