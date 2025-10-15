# app.py
import streamlit as st
from llm import context_aware_chatbot
from vector_builder import load_or_create_faiss_index
import os
from db_manager import file_monitor_db
import sqlite3

# SQLite connection (reuse existing DB)
conn = sqlite3.connect("storage/file_monitor.db", check_same_thread=False)
cursor = conn.cursor()

def get_all_files():
    cursor.execute("SELECT file_name FROM files")
    return [row[0] for row in cursor.fetchall()]
# Page config
st.set_page_config(
    page_title="PDF Chat Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "index_created" not in st.session_state:
    st.session_state.index_created = os.path.exists("faiss_indexes/index.faiss")  # ✅ Check file

if "document_name" not in st.session_state:
    st.session_state.document_name = None

# Sidebar
with st.sidebar:

    st.title("📚 Knowledge Base")
    st.subheader("📂 Files Available in Database")
    files_in_db = get_all_files()

    if files_in_db:
        st.markdown("**Available Files:**")
        for fname in set(files_in_db):
            st.markdown(f"- 📄 {fname}")
    else:
        st.info("No files found in the database yet.")
    

    
    st.markdown("---")
    
    # Index status
    st.subheader("📊 Status")
    if st.session_state.index_created:
        st.success("✅ Index Ready")
        if st.session_state.document_name:
            st.caption(f"📄 Document: {st.session_state.document_name}")
    else:
        st.warning("⚠️ No Index")
        st.caption("Upload a document to get started")
    
    st.markdown("---")
    
    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.caption("💡 Powered by OpenAI & LangChain")

# Main area
st.title("💬 Chat with Your Documents")

if not st.session_state.index_created:
    st.info("👈 Please upload a PDF document in the sidebar to get started!")
    st.markdown("""
    ### How to use:
    1. 📤 Upload a PDF document using the sidebar
    2. 🔨 Click "Process Document" to create the index
    3. 💬 Start asking questions about your document!
    """)
else:
    st.markdown("Ask me anything about your documents!")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question..."):
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display assistant response with streaming
    with st.chat_message("assistant"):
        try:
            response = st.write_stream(context_aware_chatbot(prompt))
            
            # Save to history
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            error_message = f"⚠️ Error: {str(e)}\n\nPlease make sure you've uploaded and processed a document first."
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})