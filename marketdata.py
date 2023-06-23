from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import pandas as pd
import os
loader = CSVLoader(file_path='news/Historical_Stock_Prices.csv')  
index_creator = VectorstoreIndexCreator()
docsearch = index_creator.from_loaders([loader])
chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.vectorstore.as_retriever(), input_key="question")
query = "how many ticker symbol are there?"
response = chain({"question": query})
print(response['result'])
query = "provide all unique ticker?"
response = chain({"question": query})
print(response['result'])
query = "which company is this?"
response = chain({"question": query})
print(response['result'])
query = "what was the highest stock price as per sheet?"
response = chain({"question": query})
print(response['result'])
query = "which date was it? give in mm dd yyyy format"
response = chain({"question": query})
print(response['result'])