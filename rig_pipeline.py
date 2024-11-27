from zenml import step, pipeline
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
from typing import Tuple, List
from transformers import pipeline as hf_pipeline
import cohere
import google.generativeai as genai
from langchain.vectorstores import FAISS
import faiss
import os
import streamlit as st

genai.configure(api_key="AIzaSyB_1802hLH-rIpwx8EAC6uQwcYcxnt0bNM")

@step
def preprocess_text(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400, separators=["\n\n", "\n", ".", " "])
    return text_splitter.split_text(raw_text)

@step
def extract_pdf_text(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() + "\n"
    return full_text

@step
def create_faiss_index(text_chunks):
    co = cohere.Client("A1m977Y7aoGcEz1IXgGIeRD7Mcbvq1eHAjQoQ5qf")
    batch_size = 96
    all_embeddings = []
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i + batch_size]
        response = co.embed(texts=batch, model="embed-multilingual-v3.0", input_type="search_document")
        all_embeddings.extend(response.embeddings)
    embeddings = np.array(all_embeddings, dtype=np.float32)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return embeddings, text_chunks, index

@step
def retrieve_relevant_chunks(question, index_data, top_k=12):
    embeddings, text_chunks, index = index_data
    co = cohere.Client("A1m977Y7aoGcEz1IXgGIeRD7Mcbvq1eHAjQoQ5qf")
    query_response = co.embed(texts=[question], model="embed-multilingual-v3.0", input_type="search_query")
    query_embedding = np.array([query_response.embeddings[0]], dtype=np.float32)
    k = max(min(top_k, len(text_chunks)), 5)
    scores, indices = index.search(query_embedding, k)
    formatted_chunks = []
    for idx in indices[0]:
        if idx < len(text_chunks):
            chunk = text_chunks[idx].strip()
            if chunk:
                formatted_chunks.append(chunk)
    if not formatted_chunks:
        return "Aucun contexte pertinent trouvé."
    return "\n\n".join(formatted_chunks)

@step
def generate_answer(question, context):
    model = genai.GenerativeModel("gemini-pro")
    prompt = f"Tu es un assistant juridique expert spécialisé dans l'analyse des lois sur l'enseignement supérieur au Maroc. Contexte: {context[:10000]} Question: {question}"
    try:
        generation_config = {"temperature": 0.7, "top_p": 0.8, "top_k": 40, "max_output_tokens": 2048}
        response = model.generate_content(prompt, generation_config=generation_config)
        return response.parts[0].text.strip() if response.parts else "Je n'ai pas pu générer une réponse appropriée."
    except Exception as e:
        return "Je m'excuse, je n'ai pas pu accéder aux informations demandées."

@pipeline
def rag_pipeline(pdf_path, question):
    raw_text = extract_pdf_text(pdf_path)
    text_chunks = preprocess_text(raw_text)
    index_data = create_faiss_index(text_chunks)
    relevant_context = retrieve_relevant_chunks(question, index_data)
    answer = generate_answer(question, relevant_context)
    return answer
