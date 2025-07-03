
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Configuration
DATA_PATH = "data/" # Directory where your PDF files are located
VECTOR_DB_PATH = "./chroma_db" # Directory to store the vector database
LLM_MODEL = "gemma3:4b" # Local LLM model for generating answers
EMBEDDING_MODEL = "nomic-embed-text:latest" # Local embedding model for generating embeddings

# 2. Load Documents
def load_documents(data_path):
    loader = PyPDFDirectoryLoader(data_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")
    return documents

# 3. Split Text
#
# chunk_size:
#   Defines the maximum size of each text chunk in characters.
#   Larger chunks can provide more context but must fit
#   within the LLM's context window.
# chunk_overlap:
#   Specifies the number of characters that overlap between
#   consecutive chunks. This helps maintain context across
#   chunk boundaries and prevents loss of information.
# add_start_index:
#   When True, a metadata field containing the starting index
#   of the chunk within the original document is added. Useful
#   for traceability and debugging.
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks.")
    return chunks

# 4. Create Embeddings and Vector Store
#
# documents:
#   The list of text chunks to be converted into embeddings.
# embedding_model:
#   The local embedding model used to convert text chunks into
#   numerical vector representations. This model should be
#   compatible with Ollama.
# vector_db_path:
#   The directory where the vector store will be persisted.
#   This allows for efficient storage and retrieval of embeddings.
#   The vector store can be queried later to find relevant chunks
#   based on user queries.
def create_vector_store(chunks, embedding_model, vector_db_path):
    print("Creating embeddings and vector store...")
    embeddings = OllamaEmbeddings(model=embedding_model)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=vector_db_path
    )
    print("Vector store created and persisted.")
    return vector_store

# 5. Initialize LLM
def initialize_llm(llm_model):
    print(f"Initializing LLM: {llm_model}")
    llm = OllamaLLM(model=llm_model)
    return llm

# 6. Build the RAG Chain
def build_rag_chain(vector_store, llm):
    retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 relevant chunks

    prompt_template = """
    You are an AI assistant for question-answering over documents.
    Use the following retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.

    Context: {context}
    Question: {question}
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("RAG chain built.")
    return rag_chain

# Main execution flow
def main():
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"Please place your PDF files in the '{DATA_PATH}' directory.")
        return

    documents = load_documents(DATA_PATH)
    if not documents:
        print("No PDF documents found. Please add PDFs to the 'data/' directory.")
        return

    chunks = split_documents(documents)
    vector_store = create_vector_store(chunks, EMBEDDING_MODEL, VECTOR_DB_PATH)
    llm = initialize_llm(LLM_MODEL)
    rag_chain = build_rag_chain(vector_store, llm)

    print("\nAI Agent is ready! You can now ask questions about your PDFs.")
    print("Type 'exit' to quit.")

    while True:
        user_query = input("\nYour question: ")
        if user_query.lower() == 'exit':
            print("Exiting agent. Goodbye!")
            break

        try:
            response = rag_chain.invoke(user_query)
            print(f"\nAgent's Answer: {response}")
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please ensure Ollama is running and the models are downloaded.")

if __name__ == "__main__":
    main()