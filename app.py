import streamlit as st
import os
import tempfile
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

st.set_page_config(page_title="RAG Chatbot 📄", page_icon="🤖")
st.title("Chat con tu PDF (RAG) 🤖")

# --- Configuración en Barra Lateral ---
with st.sidebar:
    st.header("Configuración")
    api_key = st.text_input("Introduce tu Groq API Key:", type="password")
    uploaded_file = st.file_uploader("Sube un archivo PDF", type="pdf")
    st.info("Obtén tu clave en [console.groq.com](https://console.groq.com/keys)")

# --- Función para procesar el PDF ---
def process_pdf(file, _embeddings):
    # Guardar archivo temporalmente para que PyMuPDF pueda leerlo
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_path = tmp_file.name

    loader = PyMuPDFLoader(tmp_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    
    vectorstore = FAISS.from_documents(documents=splits, embedding=_embeddings)
    os.remove(tmp_path) # Limpiar archivo temporal
    return vectorstore

# --- Lógica Principal ---
if api_key and uploaded_file:
    try:
        # 1. Cargar Embeddings (se cachean para no recargar)
        @st.cache_resource
        def load_embeddings():
            return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        
        embeddings = load_embeddings()

        # 2. Procesar el PDF y crear el Vector Store
        vectorstore = process_pdf(uploaded_file, embeddings)
        retriever = vectorstore.as_retriever()

        # 3. Configurar el LLM y la cadena de RAG
        llm = ChatGroq(groq_api_key=api_key, model="llama-3.3-70b-versatile")
        
        system_prompt = (
            "Usa el siguiente contexto para responder la pregunta. "
            "Si no sabes la respuesta, di que no lo sabes. "
            "\n\n"
            "{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        # 4. Interfaz de Chat
        user_input = st.text_input("Haz una pregunta sobre el documento:")
        if user_input:
            with st.spinner("Pensando..."):
                response = rag_chain.invoke({"input": user_input})
                st.markdown("### Respuesta:")
                st.write(response["answer"])

    except Exception as e:
        st.error(f"Ocurrió un error: {e}")
else:
    st.warning("Por favor, introduce tu API Key y sube un archivo PDF para comenzar.")
