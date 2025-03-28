from vistas.resumen_agent import MedicalAgent
import time

def initialize_medical_agent():
    return MedicalAgent(csv_path="data/resumen_notas.csv")

def process_patient_query(agent, query):
    result = agent.process_patient(query, max_revisions=3, thread={"thread_id":"8"})
    return result

def save_txt(content, folder):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{folder}/resumen_{timestamp}.txt"
    with open(filename, "w", encoding="utf-8") as file:
        file.write(content)
    return filename