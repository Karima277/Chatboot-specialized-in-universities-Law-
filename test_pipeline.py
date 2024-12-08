import pytest
from unittest.mock import MagicMock, patch
from rag_pipeline import preprocess_text, extract_pdf_text, create_faiss_index, retrieve_relevant_chunks, generate_answer

# Mock Constants
PDF_PATH = "/kaggle/input/loi-n-01/loi-n-01-00-portant-organisation-de-lenseignement-suprieur.pdf"
QUESTION = "Quelles sont les lois sur l'enseignement supérieur au Maroc ?"
MOCK_TEXT = "Loi 00.01 sur l'enseignement supérieur au Maroc.\nLoi 01.00 pour la recherche scientifique."
MOCK_TEXT_CHUNKS = ["Loi 00.01 sur l'enseignement supérieur.", "Loi 01.00 pour la recherche scientifique."]
MOCK_EMBEDDINGS = [[0.1, 0.2, 0.3]] * len(MOCK_TEXT_CHUNKS)
MOCK_INDEX = MagicMock()
MOCK_CONTEXT = "Loi 00.01 sur l'enseignement supérieur."

# Test `extract_pdf_text`
@patch("rag_pipeline.PyPDF2.PdfReader")
def test_extract_pdf_text(mock_pdf_reader):
    mock_reader_instance = MagicMock()
    mock_reader_instance.pages = [MagicMock(extract_text=lambda: "Page 1 text"), MagicMock(extract_text=lambda: "Page 2 text")]
    mock_pdf_reader.return_value = mock_reader_instance

    result = extract_pdf_text(PDF_PATH)
    assert result == "Page 1 text\nPage 2 text\n", "PDF extraction failed."

# Test `preprocess_text`
def test_preprocess_text():
    result = preprocess_text(MOCK_TEXT)
    assert isinstance(result, list), "Preprocessed text is not a list."
    assert len(result) > 0, "Preprocessed text is empty."

# Test `create_faiss_index`
@patch("rag_pipline.cohere.Client")
@patch("rag_pipline.faiss.IndexFlatIP")
def test_create_faiss_index(mock_faiss, mock_cohere):
    mock_cohere_instance = MagicMock()
    mock_cohere_instance.embed.return_value = MagicMock(embeddings=MOCK_EMBEDDINGS)
    mock_cohere.return_value = mock_cohere_instance
    mock_index_instance = MagicMock()
    mock_faiss.return_value = mock_index_instance

    embeddings, text_chunks, index = create_faiss_index(MOCK_TEXT_CHUNKS)
    assert len(embeddings) == len(MOCK_TEXT_CHUNKS), "Embeddings do not match text chunks."
    assert text_chunks == MOCK_TEXT_CHUNKS, "Text chunks are incorrect."
    assert index == mock_index_instance, "FAISS index was not created."

# Test `retrieve_relevant_chunks`
def test_retrieve_relevant_chunks():
    MOCK_INDEX.search.return_value = ([], [[0]])
    index_data = (MOCK_EMBEDDINGS, MOCK_TEXT_CHUNKS, MOCK_INDEX)
    result = retrieve_relevant_chunks(QUESTION, index_data)
    assert MOCK_CONTEXT in result, "Relevant context retrieval failed."

# Test `generate_answer`
@patch("rag_pipline.genai.GenerativeModel")
def test_generate_answer(mock_genai_model):
    mock_model_instance = MagicMock()
    mock_model_instance.generate_content.return_value = MagicMock(parts=[MagicMock(text="Voici votre réponse.")])
    mock_genai_model.return_value = mock_model_instance

    response = generate_answer(QUESTION, MOCK_CONTEXT)
    assert "Voici votre réponse." in response, "Generated answer is incorrect."
