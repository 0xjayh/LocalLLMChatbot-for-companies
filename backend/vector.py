from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

#extract the data from the csv using pandas
df = pd.read_csv("possible_solutions.csv")

#initialize the embedding model
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

#instantiate the db location and create one if none exists
db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

#Point the embedding model to iterate through the rows of the document
if add_documents:
    documents = []
    ids = []
    for index, row in df.iterrows():
        document =  Document(
            page_content=row['Title'] +""+ row['Solution'],
            metadata={},
            id=str(index)
        )
        ids.append(str(index))
        documents.append(document)


#vector stor configuratione
vector_store = Chroma(
    collection_name = "company_response_bank",
    persist_directory = db_location,
    embedding_function = embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

#Set up the retriever to integrate with the vector database
retriever = vector_store.as_retriever(
    search_kwargs={"k": 1}
)