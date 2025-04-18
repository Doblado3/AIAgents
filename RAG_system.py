from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.passthrough import RunnableAssign
from langchain_community.document_transformers import LongContextReorder
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
from typing import List

from concurrent.futures import ThreadPoolExecutor


# Modelo 1: Mistral-Nemo 12b
# Embedding model 1: nomic embedding

instruct_llm = ChatOllama(model = "llama3.1:latest", temperature=0.2)

embedder = OllamaEmbeddings(model = "nomic-embed-text:latest") # No cambiar modelo de embeddings a no ser que cambiemos la base vectorial

data_base = FAISS.load_local("conversation_index", embedder, allow_dangerous_deserialization=True)
#docus = data_base.similarity_search("Testing the index")
#print(docus[0].page_content)

# ----- RAG CHAIN -----

llm = instruct_llm | StrOutputParser()


chat_prompt = ChatPromptTemplate.from_template(
    "You are a medical conversation history chatbot. Help the user as they ask questions about the conversation records."
    " User messaged just asked you a question: {input}\n\n"
    " The following information may be useful for your response: "
    " Document Retrieval:\n{context}\n\n"
    " (Answer only from retrieval. Make your response conversational)"
    "\n\nUser Question: {input}"
)

def output_puller(inputs):
    """"Output generator. Useful if your chain returns a dictionary with key 'output'"""
    if isinstance(inputs, dict):
        inputs = [inputs]
    for token in inputs:
        if token.get('output'):
            yield token.get('output')



#Cadena de enriquecimiento del input
query_reformulation = ChatPromptTemplate.from_template(
    "You are an expert writing prompts to Large Language Models."
    "User messaged just asked you a question: {input}\n\n"
    "First, identify the key context concepts usefull for the query from the conversation track."
    "Second, enrich the query based on your knowleadge and the findings you made."
    "Finally, synthesize your findings into a final query that covers the initial user concerns."
    "Use ONLY information from the provided CONTEXT. If the information is not sufficient, acknowledge the limitations of your response."
    "Do not include any commentary like 'Here is your response."
    "\n\nExample:"
    "User: find similar patients"
    "assistant: Find patients with similars to the one we just analyzed, which name was Jone Doe and his medical conditions were..."
)

enrich_chain = query_reformulation | llm  

# Cadena de extracción de contexto       
long_reorder = RunnableLambda(LongContextReorder().transform_documents) 
context_getter = itemgetter('input') | data_base.as_retriever() | long_reorder
retrieval_chain = {'input' : (lambda x: x)} | RunnableAssign({'context' : context_getter})      

# Cadena de generación de la respuesta
generator_chain = chat_prompt | llm 
generator_chain = {"output" : generator_chain } | RunnableLambda(output_puller)

rag_chain = enrich_chain | retrieval_chain | generator_chain

for token in rag_chain.stream("Can you refresh my memory about the Vietnamese female patient?"):
    print(token, end="")
    
""" Lista de posibles TODOS:

    -Evaluar este RAG con preguntas respuestas sintéticas
    -Crear un chat explícito que decida si accede a info de contexto o no
    -Crear un sistema que genere resúmenes en base a estas conversaciones
    -Conectar a la API de Pubmed y hacer un asistente de IA
    -Controlar el uso de tokens
    -Enriquecer la query del usuario(... en proceso ...)
    -Estudiar mejora en los tiempos de respuesta
    
"""
