from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter

from langchain_community.vectorstores import FAISS
#from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

import os
import tarfile


""" Este archivo contiene el código para cargar y almacenar los embeddings del CSV de conversaciones en un base de datos vectorial FAISS"""
""" Está en un archivo aparte ya que ejecutarlo cada vez que llamamos a un modelo no es viable para los tiempos de respuesta"""

# Cargamos los documentos indicando el ID en los metadatos y estructurando el contenido por Columnas


# Datos: https://github.com/abachaa/MTS-Dialog/blob/main/Augmented-Data/MTS-Dialog-Augmented-TrainingSet-3-FR-and-ES-3603-Pairs-final.csv


loader = CSVLoader(file_path="data/taining_set_limpio.csv", metadata_columns=['section_header','section_text'], csv_args={
        "delimiter": ",",
        "quotechar": '"',
        "fieldnames": ["ID", "section_header", "section_text", "dialogue"],
    })

data = loader.load()
data = data[1:]


embedder = OllamaEmbeddings(model = "nomic-embed-text:latest")

#model_name = "sentence-transformers/all-mpnet-base-v2"
#embedder = HuggingFaceEmbeddings(model_name = model_name)

# Vamos a usar como base vectorial FAISS(facebook), automatiza del proceso de generar los embeddings para los docus,
# almacenarlos en una base de datos, generar los embeddings para las querys y buscar por similitud.

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=80,
)

docs_split = text_splitter.split_documents(data)

datastore = FAISS.from_documents(docs_split, embedding=embedder)

datastore.save_local("training_set_index_character_512_nomic")

with tarfile.open("training_set_index_character_512_nomic.tgz", "w:gz") as tar:
    tar.add("training_set_index_character_512_nomic", arcname=os.path.basename("training_set_index_character_512_nomic"))