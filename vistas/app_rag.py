import streamlit as st
from utils.rag_agent import RAGChatbot

# Set title for the page
st.title("Sección de consultas 👨🏼‍💻")

# Initialize session state
if "chatbot" not in st.session_state:
    st.session_state.chatbot = RAGChatbot()
    st.session_state.messages = []
    


# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Get user query
user_query = st.chat_input("¿Qué información desea consultar sobre los resumenes médicos?")

if user_query:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_query)
    
    # Process the query
    with st.spinner("Procesando consulta..."):
        response = st.session_state.chatbot.process_query(user_query)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Display assistant response
    with st.chat_message("assistant"):
        st.write(response)