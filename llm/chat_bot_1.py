from llm.bot import load_gpt_llm
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from loging import setup_logger
from config import load_open_ai_api_config

config_values = load_open_ai_api_config()
key = config_values["api_key"]
logger = setup_logger('faiss_maker')
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=key)

llm = load_gpt_llm()

# Define State
class ChatState(TypedDict):
    messages: Annotated[list, operator.add]
    context: str
    sources: list

# Node function
def retrieve_and_generate(state: ChatState):
    messages = state["messages"]
    last_message = messages[-1].content
    
    # Retrieve from FAISS
    index_dir = "faiss_indexes"
    vector_store = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(last_message, k=3)
    
    # Extract context and sources
    context = "\n\n".join([doc.page_content for doc in docs])
    sources = [doc.metadata.get('filename', 'Unknown') for doc in docs]
    unique_sources = list(set(sources))
    
    # Build prompt with history
    system_msg = SystemMessage(content="You are to generate the relevant answer based on the context for the input question")
    context_msg = SystemMessage(content=f"Context:\n{context}")
    
    # Prepare messages for LLM (include history)
    llm_messages = [system_msg, context_msg] + messages
    

    final_answer= llm.invoke(llm_messages)
    
    # Add sources to response
    response_with_sources = f"{final_answer.content}\n\n**Sources:** {', '.join(unique_sources)}"
    
    return {
        "messages": [AIMessage(content=response_with_sources)],
        "context": context,
        "sources": unique_sources
    }

# Build graph
workflow = StateGraph(ChatState)
workflow.add_node("rag_agent", retrieve_and_generate)
workflow.set_entry_point("rag_agent")
workflow.add_edge("rag_agent", END)

# Compile with memory
memory = InMemorySaver()
app = workflow.compile(checkpointer=memory)

# Usage function
def context_aware_chatbot(question, thread_id="default"):
    config = {"configurable": {"thread_id": thread_id}}
    
    # Use stream_mode="messages" for token-by-token streaming
    for message_chunk, metadata in app.stream(
        {"messages": [HumanMessage(content=question)]}, 
        config,
        stream_mode="messages"  # ✅ This enables token streaming
    ):
        # Only yield content chunks (skip empty ones)
        if message_chunk.content:
            yield message_chunk.content


