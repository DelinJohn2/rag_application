import os
from langchain_openai import OpenAIEmbeddings
from typing import List
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from loging import setup_logger
from config import load_open_ai_api_config
from .pdf_parser import load_pdf_pages

config_values = load_open_ai_api_config()
key = config_values["api_key"]
logger = setup_logger('faiss_maker')
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=key)

def load_or_create_faiss_index( document,filename,batch_size: int = 100) -> str:
    docs=load_pdf_pages(document,filename)
    """
    Load existing FAISS index or create a new one.
    Add documents in batches to prevent memory overload.
    """
    try:
        index_dir = "faiss_indexes"  

        if os.path.exists(index_dir):
            way='loaded'

            vector_store = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
            logger.info("Loaded existing FAISS index.")
        else:
            way='created'

            dim = len(embeddings.embed_query("hello world"))  # Embedding dimension
            index = faiss.IndexFlatL2(dim)
            vector_store = FAISS(
                embedding_function=embeddings,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )
            logger.info("Created new FAISS index.")

        # --- Add documents in batches ---
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            vector_store.add_documents(batch)
            logger.info(f"Added batch {i//batch_size + 1} containing {len(batch)} documents")

        # Save the FAISS index
        vector_store.save_local(index_dir)
        logger.info(f"FAISS index saved as: {index_dir}")
        success_dict={"return_message":f"Successfully {way} FAISS index","state":True}
        return success_dict

    except Exception as e:
        logger.error(f"Failed to create FAISS index: {str(e)}")
        error_message={"return_message":f"Failed to create FAISS index: {str(e)}","state":False}
        return error_message
