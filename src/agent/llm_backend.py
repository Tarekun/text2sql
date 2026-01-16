from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import Config

def instantiate_llm(config: Config):
    model = ChatGoogleGenerativeAI(
        model=config.model_name,
        temperature=config.model_settings.temperature,
        project=config.gcp_project,
    )

    return model
