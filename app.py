import streamlit as st
import os
from pathlib import Path
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="SafeBank RAG Assistant", layout="centered")
st.title("📄 Asistente Virtual con RAG")

# --- BARRA LATERAL: Configuración ---
with st.sidebar:
    st.header("Configuración")
    api_key = st.text_input("Introduce tu Groq API Key", type="password")
    model_id = st.selectbox("Modelo", ["llama-3.3-70b-versatile", "llama3-8b-8192"])
    temperature = st.slider("Temperatura", 0.0, 1.5, 0.7, step=0.1)
    
    uploaded_file = st.file_uploader("Sube un manual (PDF)", type="pdf")

# --- FUNCIONES CORE ---
def process_pdf(file):
    # Guardar temporalmente para que PyMuPDF pueda leerlo
    temp_path = Path("temp_manual.pdf")
    with open(temp_path, "wb") as f:
        f.write(file.getvalue())
    
    # 1. Cargar y Extraer
    loader = PyMuPDFLoader(str(temp_path))
    docs = loader.load()
    content = "\n".join([page.page_content for page in docs])
    
    # 2. Splitter (Chunks)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(content)
    
    # 3. Embeddings y Vectorstore
    embedding_model = "BAAI/bge-large-en-v1.5"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    
    return vectorstore

# --- FLUJO PRINCIPAL ---
if api_key and uploaded_file:
    os.environ["GROQ_API_KEY"] = api_key
    
    # Procesar el archivo una sola vez
    if "vectorstore" not in st.session_state:
        with st.spinner("Indexando documento..."):
            st.session_state.vectorstore = process_pdf(uploaded_file)
            st.success("¡Documento indexado con éxito!")

    # Configuración de la cadena RAG
    llm = ChatGroq(model=model_id, temperature=temperature)
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 6})
    
    system_prompt = """You are a helpful virtual assistant answering questions about a company's services.
    Use the following bits of retrieved context to answer the question.
    If you don't know the answer, just say you don't know. Keep your answer concise. \n\n"""

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Question: {input}\n\n Context: {context}"),
    ])

    chain = (
        {"context": retriever, "input": RunnablePassthrough()}
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    # Chat
    query = st.text_input("Haz una pregunta sobre el manual:")
    if query:
        with st.spinner("Pensando..."):
            response = chain.invoke(query)
            st.markdown("### Respuesta:")
            st.write(response)

elif not api_key:
    st.info("Por favor, introduce tu API Key de Groq en la barra lateral para comenzar.")
