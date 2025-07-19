from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from vector import build_chroma_index

key1=''
embedding = OpenAIEmbeddings(openai_api_key=key1)

# Load the vector store
vectordb = build_chroma_index(embedding)

# Set up retrieval + LLM
retriever = vectordb.as_retriever(search_type="similarity", k=3)
llm = ChatOpenAI(openai_api_key=key1,temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True  # <-- to get source info
)

# Ask a medical question
query = "What are the symptoms of diabetes?"
response = qa_chain(query)

print("\n=== Answer ===")
print(response["result"])

print("\n=== Sources Used ===")
for doc in response["source_documents"]:
    print(f"File: {doc.metadata['source']} | Page: {doc.metadata['page']}")

