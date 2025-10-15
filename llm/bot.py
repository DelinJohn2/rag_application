from config import  load_open_ai_api_config
from langchain.chat_models import init_chat_model
from loging.logger import setup_logger


logger = setup_logger("LLMLoader")


def load_gpt_llm():
    try:
        logger.info("Loading OpenAI GPT LLM...")
        config_files = load_open_ai_api_config()

        # os.environ["OPENAI_API_KEY"] = config_files["api_key"]

        llm = init_chat_model(
            model=config_files["model"],
            model_provider=config_files["provider"],
            api_key=config_files["api_key"],
        )
        logger.info("GPT LLM loaded successfully: model=%s", config_files["model"])
        return llm
    except Exception:
        logger.exception("Failed to load GPT LLM")
        raise
