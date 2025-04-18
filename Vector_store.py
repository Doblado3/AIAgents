from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os
import tarfile


""" Este archivo contiene el código para cargar y almacenar los embeddings del CSV de conversaciones en un base de datos vectorial FAISS"""
""" Está en un archivo aparte ya que ejecutarlo cada vez que llamamos a un modelo no es viable para los tiempos de respuesta"""

# Cargamos los documentos indicando el ID en los metadatos y estructurando el contenido por Columnas


# Datos: https://github.com/abachaa/MTS-Dialog/blob/main/Augmented-Data/MTS-Dialog-Augmented-TrainingSet-3-FR-and-ES-3603-Pairs-final.csv


loader = CSVLoader(file_path="data/MTS-Dialog-Augmented-TrainingSet-3-FR-and-ES-3603-Pairs-final.csv", source_column="ID", csv_args={
        "delimiter": ",",
        "quotechar": '"',
        "fieldnames": ["ID", "section_header", "section_text", "dialogue"],
    })

data = loader.load()

"""
# --- TODO: ¿Funcionará mejor si lo cargamos como etiquetas HTML para los modelos? ---
loader = UnstructuredCSVLoader(
    file_path="data/MTS-Dialog-TrainingSet.csv", mode="elements"
)
docs = loader.load()

print(docs[0].metadata["text_as_html"][:50])
"""

#print("Número de Documents extraidos:", len(data)) # 3604 ID's
#print("Número de Tokens aproximados por documento:", len(data[1].page_content)) # Más o menos cabrían unos 4 Documents en la ventana de contexto de un modelo medio

embedder = OllamaEmbeddings(model = "nomic-embed-text:latest")

#model_name = "sentence-transformers/all-mpnet-base-v2"
#embedder = HuggingFaceEmbeddings(model = model_name)

# Vamos a usar como base vectorial FAISS(facebook), automatiza del proceso de generar los embeddings para los docus,
# almacenarlos en una base de datos, generar los embeddings para las querys y buscar por similitud.

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=128,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", ";", ",", " ", ""],
)

docs_split = text_splitter.split_documents(data)

datastore = FAISS.from_documents(data, embedding=embedder)

datastore.save_local("conversation_index")

with tarfile.open("conversation_index.tgz", "w:gz") as tar:
    tar.add("conversation_index", arcname=os.path.basename("conversation_index"))