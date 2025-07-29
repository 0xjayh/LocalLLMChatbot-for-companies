from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_ollama.llms import (OllamaLLM)
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever


model =  OllamaLLM(model="mistral")

template = """
You are an expert in proferring solution to certain domestic problems

Here are some relevant documentations: {answer_bank}
Here is the problem to be solved: {question}
"""

prompt_template = ChatPromptTemplate.from_template(template)
chain = prompt_template | model

while True:
    print("\n\n------------------")
    question = input("Ask your question(q to quit): ")

    print("\n\n")
    if question == "q":
        break

    answer_bank = retriever.invoke(question)
    result = chain.invoke({"answer_bank": answer_bank, "question" : question})
    print(result)