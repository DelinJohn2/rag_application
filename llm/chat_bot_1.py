from .bot import load_gpt_llm
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
    
    # Generate response
    response = llm.invoke(llm_messages)
    
    # Add sources to response
    response_with_sources = f"{response.content}\n\n**Sources:** {', '.join(unique_sources)}"
    
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
    """
    Non-streaming version
    """
    result = app.invoke(
        {"messages": [HumanMessage(content=question)]},
        {"configurable": {"thread_id": thread_id}}
    )
    return result["messages"][-1].content

# Streaming version
def context_aware_chatbot_stream(question, thread_id="default"):
    """
    Streaming version - yields chunks
    """
    index_dir = "faiss_indexes"
    vector_store = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(question, k=3)
    
    context = "\n\n".join([doc.page_content for doc in docs])
    sources = [doc.metadata.get('filename', 'Unknown') for doc in docs]
    unique_sources = list(set(sources))
    
    # Get conversation history
    config = {"configurable": {"thread_id": thread_id}}
    state = app.get_state(config)
    history_messages = state.values.get("messages", []) if state.values else []
    
    # Build prompt with history
    system_msg = f"""You are to generate the relevant answer based on the context for the input question
    
Context:
{context}"""
    
    messages = [SystemMessage(content=system_msg)] + history_messages + [HumanMessage(content=question)]
    
    # Stream response
    full_response = ""
    for chunk in llm.stream(messages):
        full_response += chunk.content
        yield chunk.content
    
    # Yield sources
    sources_text = f"\n\n**Sources:** {', '.join(unique_sources)}"
    yield sources_text
    
    # Save to memory after streaming
    app.update_state(
        config,
        {
            "messages": [
                HumanMessage(content=question),
                AIMessage(content=full_response + sources_text)
            ]
        }
    )