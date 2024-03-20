from langchain_community.llms import Ollama
from langchain_community.document_loaders import word_document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import os

model = Ollama(base_url='http://localhost:11434', model="mistral")

loader = word_document.Docx2txtLoader(r"./data/Project 2 Report.docx")
data = loader.load()

text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

oembed = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text")
vectorstore = Chroma.from_documents(documents=all_splits, embedding=oembed)

questions=["What was the intent of the project that's detailed in the Project 2 Report?",
           "What was the result for TSM and why does it differ from MSC?",
           "Can you perform an uncertainty analysis on measuring a 20 foot table with a tape measure using the Project 2 Report as a reference?"]

for question in questions:
    docs = vectorstore.similarity_search(question)

    qachain=RetrievalQA.from_chain_type(model, retriever=vectorstore.as_retriever())
    response = qachain.invoke({"query": question})

    result = response['result']

    text = f'Question: {question}\n\nModel Response: {result}'

    concount = len(os.listdir(r'./conversations'))

    with open(rf'./conversations/conversation{concount}.txt', 'w') as f:
        f.write(text)

    print(f'conversation{concount}.txt Saved')