# Streamlit App for Legal Assistant
import streamlit as st
from zenml import step, pipeline
import PyPDF2
from typing import Tuple, List
import numpy as np
import faiss
import os
from rag_pipeline import (
    extract_pdf_text,
    preprocess_text,
    create_faiss_index,
    retrieve_relevant_chunks,
    generate_answer
)

st.set_page_config(
    page_title="Assistant Juridique - Loi sur l'Enseignement Supérieur",
    page_icon="📚",
    layout="wide"
)

st.markdown("""
<style>
    .chat-container {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e6f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .assistant-message {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'index_data' not in st.session_state:
    st.session_state.index_data = None

def initialize_rag_system(pdf_path):
    try:
        raw_text = extract_pdf_text.entrypoint(pdf_path)
        text_chunks = preprocess_text.entrypoint(raw_text)
        index_data = create_faiss_index.entrypoint(text_chunks)
        st.session_state.index_data = index_data
        return True
    except Exception as e:
        st.error(f"Erreur lors de l'initialisation: {str(e)}")
        return False

with st.sidebar:
    st.title("🤖 Assistant Juridique")
    st.markdown("""
    ### À propos
    Cet assistant vous aide à comprendre la loi 01.00 sur l'organisation de l'enseignement supérieur au Maroc.
    
    ### Comment utiliser
    1. Posez votre question sur la loi
    2. L'assistant analysera le texte de loi
    3. Vous recevrez une réponse
    """)
    uploaded_file = st.file_uploader("Charger un nouveau PDF de loi", type="pdf")
    if uploaded_file:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getvalue())
        if initialize_rag_system("temp.pdf"):
            st.success("PDF chargé et indexé avec succès!")

st.title("💬 Assistant Juridique - Loi sur l'Enseignement Supérieur")

if st.session_state.index_data is None:
    pdf_path = "/kaggle/input/loi-n-01/loi-n-01-00-portant-organisation-de-lenseignement-suprieur.pdf"
    if os.path.exists(pdf_path):
        initialize_rag_system(pdf_path)
    else:
        st.warning("Veuillez charger un fichier PDF de la loi pour commencer.")

for message in st.session_state.messages:
    with st.container():
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">👤 Vous : {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message">🤖 Assistant : {message["content"]}</div>', unsafe_allow_html=True)

if question := st.chat_input("Posez votre question sur la loi..."):
    st.session_state.messages.append({"role": "user", "content": question})
    if st.session_state.index_data is not None:
        try:
            with st.spinner("Recherche en cours..."):
                relevant_context = retrieve_relevant_chunks.entrypoint(question=question, index_data=st.session_state.index_data, top_k=8)
                answer = generate_answer.entrypoint(question=question, context=relevant_context)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.rerun()
        except Exception as e:
            st.error(f"Erreur lors du traitement de votre question: {str(e)}")
    else:
        st.warning("Le système n'est pas encore initialisé. Veuillez charger un fichier PDF.")
if st.button("Effacer la conversation"):
    st.session_state.messages = []
    st.rerun()
