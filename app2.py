import streamlit as st
import os
import tempfile
import gc # Garbage Collection para limpiar memoria
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# --- CONFIGURACIÓN OPTIMIZADA ---
st.set_page_config(page_title="SafeBank Team AI", page_icon="🏦", layout="wide")

# --- SISTEMA DE LOGIN SIMPLE ---
def check_password():
    """Retorna True si el usuario ingresó la contraseña correcta."""
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    if st.session_state.password_correct:
        return True

    st.title("🔒 Acceso Restringido")
    pwd = st.text_input("Introduce la contraseña de acceso", type="password")
    
    if st.button("Entrar"):
        # Verifica contra los secretos o una clave por defecto
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
        gc.collect() # Fuerza limpieza de RAM
        st.success("Memoria liberada")

# --- FUNCIONES CORE (Con gestión de memoria) ---

# TTL (Time To Live): El vectorstore se borra de la RAM tras 1 hora de inactividad
# max_entries: Solo guarda los últimos 5 PDFs procesados en memoria para no saturar
@st.cache_resource(ttl="1h", max_entries=5, show_spinner=False)
def get_vectorstore_from_file(file_bytes, file_name):
    """Procesa el PDF y crea la base de datos vectorial."""
    
    # Archivo temporal seguro
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    try:
        loader = PyMuPDFLoader(tmp_path)
        docs = loader.load()
        
        # Splitter optimizado
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)
        
        # Embeddings (Usamos caché interno de HF para no recargar el modelo)
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# --- LÓGICA DE CHAT ---
st.title("💬 Chat con Documentación")

# 1. Inicializar historial de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Mostrar historial previo
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Input del usuario
if prompt := st.chat_input("¿Qué necesitas saber del documento?"):
    
    # Validar que hay archivo
    if not uploaded_file:
        st.warning("⚠️ Por favor, sube un PDF primero en la barra lateral.")
        st.stop()

    # Guardar y mostrar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 4. Generar respuesta
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            # Recuperar vectorstore (usando caché inteligente)
            vectorstore = get_vectorstore_from_file(uploaded_file.getvalue(), uploaded_file.name)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            
            # Preparar contexto
            docs = retriever.invoke(prompt)
            context_text = "\n\n".join([d.page_content for d in docs])
            
            # Prompt del sistema
            system_prompt = """Eres un asistente útil y preciso. 
            Responde basándote SOLO en el contexto proporcionado.
            Si no sabes la respuesta, dilo honestamente.
            
            Contexto: {context}
            """
            
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{question}")
            ])
            
            # Cadena
            llm = ChatGroq(model=model_option, temperature=0.5)
            chain = prompt_template | llm | StrOutputParser()
            
            # Streaming de respuesta (efecto escribir)
            full_response = ""
            for chunk in chain.stream({"context": context_text, "question": prompt}):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            
            # Guardar respuesta en historial
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
