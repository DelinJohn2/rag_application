from .bot import load_gpt_llm

from langchain_openai import OpenAIEmbeddings

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from loging import setup_logger
from config import load_open_ai_api_config

config_values = load_open_ai_api_config()
key = config_values["api_key"]
logger = setup_logger('faiss_maker')
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=key)

llm= load_gpt_llm()


def context_aware_chatbot(question):
    index_dir = "faiss_indexes"
    vector_store = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    
    # Get documents with metadata
    docs = vector_store.similarity_search(question, k=3)  # Get top 3 relevant docs
    
    # Extract context and sources
    context = "\n\n".join([doc.page_content for doc in docs])
    sources = [doc.metadata.get('filename', 'Unknown') for doc in docs]
    unique_sources = list(set(sources))  # Remove duplicates
    
    prompt = f"""You are to generate the relevant answer based on the context for the input question
    
    <question>
    {question}
    </question>
    
    <context>
    {context}
    </context>"""
    
    # Stream the chunks
    for chunk in llm.stream(prompt):
        yield chunk.content
    
    # Yield source information at the end
    yield f"\n\n**Sources:** {', '.join(unique_sources)}"