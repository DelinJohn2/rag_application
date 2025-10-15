# pdf_parser.py
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import os

def load_pdf_pages(uploaded_file, file_name):
    """
    Takes a Streamlit-uploaded PDF file (UploadedFile object)
    and returns a list of LangChain Document pages with filename in metadata.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        file_name: Name of the uploaded file
        
    Returns:
        List of LangChain Document objects with metadata
    """
    if uploaded_file is None:
        return []

    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    try:
        # Load pages using LangChain's PyPDFLoader
        loader = PyPDFLoader(tmp_file_path)
        pages = list(loader.lazy_load())
        
        # ✅ Add filename to metadata for each page
        for page in pages:
            if not hasattr(page, 'metadata') or page.metadata is None:
                page.metadata = {}
            page.metadata['filename'] = file_name
            page.metadata['file_size'] = len(uploaded_file.getvalue())
        
        return pages

    finally:
        # Clean up temporary file
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)