import streamlit as st

def load_open_ai_api_config():
    """
    Loads API configuration from Streamlit secrets or environment variables.
    Returns a dictionary with API key, model, and provider.
    """
    # Try to load from Streamlit secrets first (for Streamlit Cloud)
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        model = st.secrets.get("GPT_model")  # Default to gpt-4 if not set
        provider = st.secrets.get("GPT_model_provider")
    except (FileNotFoundError, KeyError):
        # Fallback to environment variables (for local development)
        import os
        from dotenv import load_dotenv
        
        load_dotenv(dotenv_path='.env')
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("GPT_model", "gpt-4")
        provider = os.getenv("GPT_model_provider", "openai")
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in secrets or environment variables.")
    
    return {
        "api_key": api_key,
        "model": model,
        "provider": provider
    }

# from dotenv import load_dotenv
# import os

# load_dotenv(dotenv_path='.env')  # Load .env file into environment variables



# def load_open_ai_api_config():
#     """
#     Loads API configuration from environment variables.
#     Returns a dictionary with API key, model, and provider.
#     """
#     api_key = os.getenv("OPENAI_API_KEY")
#     model = os.getenv("GPT_model")
#     provider = os.getenv("GPT_model_provider")

#     if not api_key:
#         raise ValueError("OPENAI_API_KEY not found in environment variables.")

#     return {
#         "api_key": api_key,
#         "model": model,
#         "provider": provider
#     }

