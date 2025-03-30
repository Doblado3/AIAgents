import streamlit as st


about_page = st.Page(
    page = "vistas/main.py",
    title = "Página principal",
    default = True
)

project_1_page = st.Page(
    page = "vistas/app_rag.py",
    title = "Chatbot"
    
)

file_browser_page = st.Page(
    page="vistas/file_browser.py",
    title="Almacen de resúmenes"
)



pg = st.navigation({
    "" : [about_page],
    "Herramientas": [project_1_page, file_browser_page]
})
pg.run()