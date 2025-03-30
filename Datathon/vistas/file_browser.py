import streamlit as st
import os
import time
import base64
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO

# Set title for the page
st.title("Archivos Guardados")

# File directory settings
txt_folder = "/Users/doblado/Datathon2025/altas"
os.makedirs(txt_folder, exist_ok=True)

# Define file operations
def read_txt_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def update_txt_file(file_path, content):
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)

def create_pdf(content, filename):
    """Generate a PDF from text content"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    flowables = []
    
    # Add a title
    title = Paragraph(f"Resumen Médico: {filename}", styles['Title'])
    flowables.append(title)
    flowables.append(Spacer(1, 20))
    
    # Add timestamp
    timestamp = Paragraph(f"Generado: {time.strftime('%d/%m/%Y %H:%M:%S')}", styles['Italic'])
    flowables.append(timestamp)
    flowables.append(Spacer(1, 20))
    
    # Split content into paragraphs and add to document
    paragraphs = content.split('\n\n')
    for para in paragraphs:
        if para.strip():
            p = Paragraph(para.replace('\n', '<br/>'), styles['Normal'])
            flowables.append(p)
            flowables.append(Spacer(1, 12))
    
    # Build PDF
    doc.build(flowables)
    pdf_data = buffer.getvalue()
    buffer.close()
    
    return pdf_data

def get_download_link(pdf_data, filename):
    """Generate a download link for the PDF"""
    b64 = base64.b64encode(pdf_data).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}.pdf">Descargar PDF</a>'
    return href

# Get all text files in the folder
files = [f for f in os.listdir(txt_folder) if f.endswith('.txt')]
files.sort(reverse=True)  # Newest first

if not files:
    st.info("No hay archivos guardados todavía.")
else:
    # Create two columns - file list and editor
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("")
        selected_file = st.selectbox(
            "Seleccione un archivo:",
            files,
            format_func=lambda x: x
        )
    
    if selected_file:
        file_path = os.path.join(txt_folder, selected_file)
        
        with col2:
            st.subheader(f"Revisa y edita posibles errores")
            
            # Read the file content
            file_content = read_txt_file(file_path)
            
            # Create a text area for editing
            edited_content = st.text_area(
                "Contenido:",
                value=file_content,
                height=400
            )
            
            # Create two columns for the buttons
            button_col1, button_col2, button_col3 = st.columns(3)
            
            with button_col1:
                # Save changes button
                if st.button("Guardar Cambios"):
                    update_txt_file(file_path, edited_content)
                    st.success("Cambios guardados exitosamente.")
            
            with button_col2:
                # Generate PDF button
                if st.button("Generar PDF"):
                    pdf_data = create_pdf(edited_content, selected_file.replace('.txt', ''))
                    st.markdown(get_download_link(pdf_data, selected_file.replace('.txt', '')), unsafe_allow_html=True)
                    st.success("PDF generado. Haga clic en el enlace para descargar.")
            
            with button_col3:
                # Delete file button
                if st.button("Eliminar Archivo", type="primary", help="Esta acción no se puede deshacer"):
                    os.remove(file_path)
                    st.warning(f"Archivo {selected_file} eliminado.")
                    st.experimental_rerun()