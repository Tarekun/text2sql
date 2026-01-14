from langchain_google_genai import ChatGoogleGenerativeAI
from src.tools import tool_list


_model = None


def instantiate_llm():
    global _model

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        project="soges-group-data-platform",
        # project="formazione-danieletarek-iaisy",
        # location="us-central1",
    )
    model_with_tools = model.bind_tools(tool_list)
    _model = model_with_tools

    return get_llm()


def get_llm() -> ChatGoogleGenerativeAI:
    global _model
    return _model  # type:ignore
