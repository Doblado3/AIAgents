import streamlit as st
from utils import resumer_call

st.set_page_config(page_title="Generador de Altas Médicas", page_icon="🩺")




# Initialize the agent
agent = resumer_call.initialize_medical_agent()

# Set up the UI
st.title("Generador de Altas Médicas 🩺")
patient_query = st.chat_input("¿Sobre qué paciente necesita el alta?")

# Process query when submitted
if patient_query is not None and patient_query.strip():
        result = resumer_call.process_patient_query(agent, patient_query)
        pdf_file = result['generate']['draft']
        filename = resumer_call.save_txt(pdf_file, "/Users/doblado/Datathon2025/altas")
        
    
        st.write("Resumen:")
        st.write(pdf_file)

