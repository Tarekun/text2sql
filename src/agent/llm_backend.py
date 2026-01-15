from langchain_google_genai import ChatGoogleGenerativeAI
from src.agent.tools import tool_list
from src.config import Config


_model = None


def instantiate_llm(config: Config):
    global _model

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=config.model_settings.temperature,
        project="soges-group-data-platform",
        # project="formazione-danieletarek-iaisy",
        # location="us-central1",
    )
    # model_with_tools = model.bind_tools(tool_list)
    # _model = model_with_tools
    _model = model

    return get_llm()


def get_llm(with_tools: bool = True) -> ChatGoogleGenerativeAI:
    global _model

    return _model.bind_tools(tool_list) if with_tools else _model  # type:ignore
