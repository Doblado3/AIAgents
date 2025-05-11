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
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.documents import Document
from langchain_core.runnables.history import RunnableWithMessageHistory



# Modelo 1: Mistral-Nemo 12b
# Embedding model 1: nomic embedding

instruct_llm = ChatOllama(model = "llama3.1:latest", temperature=0.2)

embedder = OllamaEmbeddings(model = "nomic-embed-text:latest") # No cambiar modelo de embeddings a no ser que cambiemos la base vectorial
#embedder = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2")

data_base = FAISS.load_local("conversacion/conversaciones_limpias_index_character_256_nomic", embedder, allow_dangerous_deserialization=True)


# ----- RAG CHAIN -----

llm = instruct_llm | StrOutputParser()



conversational_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a chatbot assistant. Your task is to answer the user's question using only and exclusively the Relevant Information provided below."
        "If the answer to the user's question is NOT found in the Relevant Information, please respond politely stating that you cannot find the answer with the available information."
        "Your response should be clear and conversational, based entirely on the provided context."
        "Finally, give your response in Spanish."
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
    # Puedes ajustar cómo se muestran los documentos. Aquí un ejemplo simple:
    return "\n\n".join([f"Source Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])



# Cadena de extracción de contexto, implementando Re-ranking con un modelo cross-encoder       

retriever = data_base.as_retriever(search_kwargs={"k": 20}, filter = True)
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




context_getter = itemgetter('input') | compression_retriever
retrieval_chain = {'input' : (lambda x: x)} | RunnableAssign({'context' : context_getter})      


 


base_rag_chain = (
    RunnablePassthrough.assign(
        retrieved_context=itemgetter("input")| english_translator_chain | compression_retriever, 
    ).assign(
        context_str=itemgetter("retrieved_context") | RunnableLambda(format_context_for_prompt) 
    )
    | conversational_prompt 
    | llm                 
)

conversational_rag_chain_with_memory = RunnableWithMessageHistory(
    base_rag_chain,
    get_session_history,
    input_messages_key="input",    
    history_messages_key="history", 
    #output_messages_key="answer" 
)



session_id_1 = "user123_session_y"
query1 = "Hay algún paciente de Vietnam?"

for token in conversational_rag_chain_with_memory.stream(
    {"input": query1}, # La entrada ahora es un diccionario con la clave "input"
    config={"configurable": {"session_id": session_id_1}}
):
    print(token, end="")
print("\n---")

query2 = "Puedes recordarme sus problemas de salud?" 
for token in conversational_rag_chain_with_memory.stream(
    {"input": query2},
    config={"configurable": {"session_id": session_id_1}}
):
    print(token, end="")
print("\n---")

"""
for token in rag_chain.stream("Is there any patient from Oklahoma?"):
    print(token, end="")
"""
    

