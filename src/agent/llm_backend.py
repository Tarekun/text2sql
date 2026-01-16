from langchain_google_genai import ChatGoogleGenerativeAI
from src.agent.tools import tool_list
from src.config import Config


def instantiate_llm(config: Config):
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=config.model_settings.temperature,
        project="soges-group-data-platform",
        # project="formazione-danieletarek-iaisy",
        # location="us-central1",
    )

    return model
