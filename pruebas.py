import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS


data = pd.read_csv("data/MTS-Dialog-TestSet-1-MEDIQA-Chat-2023.csv")

# Formateo de la columna ID, 3602 conversaciones
data['ID'] = range(len(data))

data.to_csv("data/test_set_limpio.csv", index=False)
