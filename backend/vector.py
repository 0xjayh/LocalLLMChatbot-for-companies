import os
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import DocArrayInMemorySearch

#df = pd.read_csv("realistic_restaurant_reviews.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

repo_file = f"Outliers.txt"

# load the txt file into the loader
loader = TextLoader(repo_file)

# Load the file into doc
docs = loader.load()

#Initialize an in memory database for vector embeddings to hold docs
db = DocArrayInMemorySearch.from_documents(
    docs,
    embeddings
)

retriever = db.as_retriever()