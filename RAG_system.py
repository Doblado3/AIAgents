import gradio as gr
import uuid 
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.passthrough import RunnableAssign
from langchain_community.document_transformers import LongContextReorder
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from operator import itemgetter
from typing import List

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.documents import Document
from langchain_core.runnables.history import RunnableWithMessageHistory

# --- Tu configuración de LangChain y RAG (sin cambios) ---

instruct_llm = ChatOllama(model = "llama3.1:latest", temperature=0.2)

embedder = OllamaEmbeddings(model = "nomic-embed-text:latest") # No cambiar modelo de embeddings a no ser que cambiemos la base vectorial

# Asegúrate de que esta ruta es correcta para tu base de datos vectorial
try:
    data_base = FAISS.load_local("conversacion/conversaciones_limpias_index_character_256_nomic", embedder, allow_dangerous_deserialization=True)
    print("Base de datos FAISS cargada correctamente.")
except Exception as e:
    print(f"Error al cargar la base de datos FAISS: {e}")
    print("Por favor, asegúrate de que la ruta 'conversacion/conversaciones_limpias_index_character_256_nomic' es correcta y que el directorio contiene los archivos necesarios.")
    # TO-DO: Manejar el error de forma correcta
    pass


# ----- RAG CHAIN -----

llm = instruct_llm | StrOutputParser()

conversational_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a conversational chat assistant. Your responses must be clear, polite, conversational, and delivered entirely in Spanish."
        "\n\n" # Adding a newline for readability in the prompt itself
        "- If the user provides a greeting or makes a general conversational comment (e.g., 'Hola', '¿Cómo estás?'), respond appropriately using your general knowledge."
        "- If the user asks a question that *could* be answered by the 'Relevant Information' provided below, you *must* answer using *only and exclusively* that information."
        "- For context-dependent questions where the answer is *not* found in the 'Relevant Information', respond politely stating that you cannot find the answer with the available information."
        "- Do not mention or include any reference to the resources or sources of the information."
    )),
    MessagesPlaceholder(variable_name="history"),
    ("human", "User Question: {input}\n\nRelevant Information:\n{context_str}")
])

translate_to_english_prompt = ChatPromptTemplate.from_template(
    "Translate the following text from Spanish to English. Output only the translated text and nothing else:\n\n{text_to_translate}"
)

english_translator_chain = translate_to_english_prompt | llm

def format_context_for_prompt(docs: List[Document]) -> str:
    if not docs:
        return "No relevant information found in documents for this query."
    return "\n\n".join([f"Source Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])

# Cadena de extracción de contexto, implementando Re-ranking con un modelo cross-encoder
retriever = data_base.as_retriever(search_kwargs={"k": 20}) # ¿filter?
model_cross = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
compressor = CrossEncoderReranker(model=model_cross, top_n=5)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

chat_history_store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in chat_history_store:
        chat_history_store[session_id] = ChatMessageHistory()
    return chat_history_store[session_id]

# Modificamos la cadena base para usar el traductor antes del retriever
# La traducción del 'input' ocurre dentro del RunnablePassthrough
base_rag_chain = (
    RunnablePassthrough.assign(
        retrieved_context=itemgetter("input") | english_translator_chain | compression_retriever,
    ).assign(
        context_str=itemgetter("retrieved_context") | RunnableLambda(format_context_for_prompt)
    )
    | conversational_prompt
    | llm
)

# Cadena principal, con memoria implementada
conversational_rag_chain_with_memory = RunnableWithMessageHistory(
    base_rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# --- Gradio ---

def chat_interface_response(message: str, history: list, session_id_state: str | None) -> list[str, str]:
    """
    Función que será llamada por Gradio para procesar el mensaje del usuario.
    Maneja la lógica de sesión y llama a la cadena RAG.

    Args:
        message: El mensaje actual del usuario.
        history: Historial de conversación proporcionado por Gradio (lista de [user, bot] pares).
                 Aunque LangChain maneja la historia internamente, Gradio lo pasa.
        session_id_state: El estado de la sesión ID mantenido por Gradio.

    Returns:
        Una lista: (respuesta del bot, nuevo estado de session_id_state).
    """
    # Si no hay un ID de sesión en el estado, generamos uno nuevo
    if session_id_state is None:
        session_id = str(uuid.uuid4())
        print(f"Nueva sesión iniciada con ID: {session_id}")
    else:
        session_id = session_id_state

    print(f"Procesando mensaje para la sesión ID: {session_id}")
    print(f"Mensaje recibido: {message}")

    response = ""
    try:
        for token in conversational_rag_chain_with_memory.stream(
            {"input": message},
            config={"configurable": {"session_id": session_id}}
        ):
             response += token
             
        print(f"Respuesta generada: {response}")
    except Exception as e:
        response = f"Lo siento, ocurrió un error al procesar tu solicitud: {e}"
        print(f"Error durante el procesamiento de LangChain: {e}")

    return [response, session_id]

iface = gr.ChatInterface(
    fn=chat_interface_response,
    additional_inputs=[
        gr.State(value=None) 
    ],
    theme="soft" 
)

if __name__ == "__main__":
    print("Lanzando interfaz Gradio...")
    iface.launch(share=False, inbrowser=True)
    print("Interfaz Gradio lanzada.")