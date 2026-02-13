import streamlit as st
import os
import tempfile
import gc 
import pytesseract
from pdf2image import convert_from_path
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="SafeBank Team AI", page_icon="🏦", layout="wide")

# --- SISTEMA DE LOGIN SIMPLE ---
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    if st.session_state.password_correct:
        return True

    st.title("🔒 Acceso Restringido")
    pwd = st.text_input("Introduce la contraseña de acceso", type="password")
    
    if st.button("Entrar"):
        correct_pwd = st.secrets.get("APP_PASSWORD", "admin123")
        if pwd == correct_pwd:
            st.session_state.password_correct = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta")
    return False

if not check_password():
    st.stop()

# --- GESTIÓN DE API KEY ---
os.environ["GROQ_API_KEY"] = st.secrets.get("GROQ_API_KEY", "")

# --- FUNCIONES AUXILIARES DE OCR ---
def perform_ocr(pdf_path):
    """Realiza OCR por lotes para no saturar la memoria RAM."""
    st.info("⚠️ Documento escaneado detectado. Procesando por partes (esto tomará tiempo)...")
    text = ""
    batch_size = 5   
    max_pages = 20   
    current_page = 1
    
    # Barra de progreso
    progress_bar = st.progress(0)
    
    while current_page <= max_pages:
        try:
            images = convert_from_path(
                pdf_path, 
                first_page=current_page, 
                last_page=min(current_page + batch_size - 1, max_pages)
            )
            
            if not images:
                break 
            
            for img in images:
                # Reducir tamaño si es muy grande para ahorrar RAM
                if img.width > 1000:
                    base_width = 1000
                    w_percent = (base_width / float(img.width))
                    h_size = int((float(img.height) * float(w_percent)))
                    img = img.resize((base_width, h_size))
                
                text += pytesseract.image_to_string(img) + "\n"
                
            progress_val = min(current_page / max_pages, 1.0)
            progress_bar.progress(progress_val)
            
            current_page += batch_size
            del images # Liberar memoria
            gc.collect()
            
        except Exception as e:
            st.warning(f"Aviso: Se detuvo el OCR en la página {current_page} ({e})")
            break
            
    progress_bar.empty()
    return text

# --- PROCESAMIENTO INTELIGENTE (CACHE) ---
@st.cache_resource(ttl="1h", max_entries=3, show_spinner=False)
def get_vectorstore_from_file(file_bytes, file_name):
    """Procesa PDF (Texto o Imagen) y crea VectorStore optimizado."""
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    try:
        # 1. Intentar leer como texto digital
        loader = PyMuPDFLoader(tmp_path)
        docs = loader.load()
        full_text = "\n".join([d.page_content for d in docs])

        # 2. Si no hay texto, usar OCR
        if len(full_text.strip()) < 50:
            ocr_text = perform_ocr(tmp_path)
            if not ocr_text.strip():
                raise ValueError("No se pudo extraer texto del archivo.")
            docs = [Document(page_content=ocr_text, metadata={"source": file_name})]
        
        # 3. Dividir en Chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_chunks = text_splitter.split_documents(docs)
        
        # 4. Crear Embeddings (MODELO LIGERO)
        # Usamos all-MiniLM-L6-v2 (80MB) en lugar de BGE (1.5GB)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # 5. Indexar por lotes (Batching)
        vectorstore = None
        batch_size = 50 # Lotes pequeños para seguridad
        
        my_bar = st.progress(0, text="Creando índice de búsqueda...")
        
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch, embeddings)
            else:
                vectorstore.add_documents(batch)
            
            progress = min((i + batch_size) / len(all_chunks), 1.0)
            my_bar.progress(progress)
            
        my_bar.empty()
        return vectorstore

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        gc.collect()

# --- BARRA LATERAL ---
with st.sidebar:
    st.title("📁 Documentación")
    uploaded_file = st.file_uploader("Cargar Manual (PDF)", type="pdf")
    
    st.markdown("---")
    st.write("⚙️ **Ajustes de IA**")
    model_option = st.selectbox("Modelo", ["llama-3.3-70b-versatile", "llama3-8b-8192"])
    
    if st.button("🧹 Limpiar Chat y Memoria"):
        st.session_state.messages = []
        st.cache_resource.clear()
        gc.collect()
        st.success("Memoria liberada")

# --- LÓGICA DE CHAT ---
st.title("💬 Chat con Documentación")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("¿Qué necesitas saber del documento?"):
    if not uploaded_file:
        st.warning("⚠️ Por favor, sube un PDF primero.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        try:
            vectorstore = get_vectorstore_from_file(uploaded_file.getvalue(), uploaded_file.name)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            
            docs = retriever.invoke(prompt)
            context_text = "\n\n".join([d.page_content for d in docs])
            
            system_prompt = """Eres un experto técnico. Responde basándote SOLO en el contexto.
            Si la información no está, dilo. Contexto: {context}"""
            
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{question}")
            ])
            
            llm = ChatGroq(model=model_option, temperature=0.3)
            chain = prompt_template | llm | StrOutputParser()
            
            full_response = ""
            for chunk in chain.stream({"context": context_text, "question": prompt}):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
