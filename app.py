import os
import getpass
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

# --- 0. Load environment variables from .env file ---
load_dotenv()

# --- 1. Set up API Key ---
try:
    # Attempt to load from environment variable first (e.g., from .env or system env)
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        # If not found, prompt the user (primarily for interactive notebook use)
        groq_api_key = getpass.getpass("Enter your Groq API Key: ")
        os.environ["GROQ_API_KEY"] = groq_api_key # Set it for the current session
except Exception as e:
    print(f"Error getting Groq API Key: {e}")
    print("Please set the GROQ_API_KEY environment variable or provide it when prompted.")
    exit()

# --- 2. Load the LLM ---
def load_llm(id_model, temperature):
  llm = ChatGroq(
      model=id_model,
      temperature=temperature,
      max_tokens=None,
      timeout=None,
      max_retries=2
  )
  return llm

id_model = "llama-3.3-70b-versatile"
temperature = 0.7
llm = load_llm(id_model, temperature)

# --- 3. Function to extract text from PDF ---
def extract_text_pdf(file_path):
  loader = PyMuPDFLoader(file_path)
  doc = loader.load()
  content = "\n".join([page.page_content for page in doc])
  return content

# --- 4. Prompt user for PDF path and load content ---
print("Please upload your PDF file (e.g., safebank-manual.pdf).")
print("If running in Colab, you might need to use `from google.colab import files` and `uploaded = files.upload()` first.")
file_path_input = input("Enter the path to your PDF file: ")

try:
    loaded_document_content = extract_text_pdf(file_path_input)
    print(f"Successfully loaded content from {file_path_input}")
except Exception as e:
    print(f"Error loading PDF file: {e}")
    exit()

# --- 5. Splitting text into chunks ---
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_text(loaded_document_content)
print(f"Number of chunks created: {len(chunks)}")

# --- 6. Embedding generation and indexing ---
embedding_model = "BAAI/bge-large-en-v1.5"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

# Create FAISS vector store
vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
print("FAISS vector store created.")

# --- 7. Configure the Retriever ---
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6}
)
print("Retriever configured.")

# --- 8. Define the RAG Prompt ---
system_prompt = """You are a helpful virtual assistant answering general questions about a company's services.
Use the following bits of retrieved context to answer the question.
If you don't know the answer, just say you don't know. Keep your answer concise. \n\n"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "Question: {input}\n\n Context: {context}"),
    ]
)
print("QA Prompt defined.")

# --- 9. Create the RAG Chain ---
chain_rag = (
    {"context": retriever, "input": RunnablePassthrough()}
    | qa_prompt
    | llm
    | StrOutputParser()
)
print("RAG chain created.")

# --- 10. Simple CLI for interaction ---
print("\nWelcome to the RAG Assistant!")
print("Type 'exit' to quit.")

while True:
    user_input_query = input("\nAsk a question: ")
    if user_input_query.lower() == 'exit':
        break
    if not user_input_query.strip():
        print("Please enter a question.")
        continue

    try:
        response = chain_rag.invoke(user_input_query)
        print("\nAssistant:", response)
    except Exception as e:
        print(f"An error occurred during response generation: {e}")
        print("Please check your API key, model, or FAISS index setup.")

print("Goodbye!")
