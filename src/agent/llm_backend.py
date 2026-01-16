from collections.abc import Sequence
from datetime import datetime
from langchain_core.prompt_values import PromptValue
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
import threading
from typing import Any
from src.config import Config


_lock = threading.Lock()


# BaseChatModel is abstract and requires implementing _generate
# for the time being im only extending ChatGoogleGenerativeAI
class LoggedChatModel(ChatGoogleGenerativeAI):
    def __init__(self, inner_llm: BaseChatModel):
        self._inner_llm = inner_llm
        self._log_path = "./logs/llm_calls.log"
        self._call_index = 0
        with open(self._log_path, "w", encoding="utf-8") as f:
            f.write("Last execution LLM convo trail\n")

    @property
    def _llm_type(self) -> str:
        return f"logged::{self._inner_llm._llm_type}"

    def _prompt_from_messages(self, messages: LanguageModelInput) -> str:
        if isinstance(messages, str):
            return messages
        elif isinstance(messages, PromptValue):
            return str(messages)
        elif isinstance(messages, Sequence):
            content = ""
            for m in messages:
                if isinstance(m, str):
                    content += m
                elif isinstance(m, list) or isinstance(m, tuple):
                    content += "".join(m)
                elif isinstance(m, dict):
                    print(m)
                    content += m["text"]
                elif isinstance(m, BaseMessage):
                    content += str(m.content)
            return content
        return "suka"

    def invoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        messages = input
        with _lock:
            self._call_index += 1
            call_number = self._call_index

        record = {
            "call_number": call_number,
            "timestamp": datetime.now().isoformat(),
            "prompt": self._prompt_from_messages(messages),
            "stop": stop,
            "config": config,
            "kwargs": kwargs,
        }
        try:
            response = self._inner_llm.invoke(
                input,
                config=config,
                stop=stop,
                **kwargs,
            )

            record["response"] = {
                "type": response.type,
                "content": response.content,
                "additional_kwargs": response.additional_kwargs,
            }

            return response

        except Exception as e:
            record["exception"] = repr(e)
            raise

        finally:
            self._write_record(record)

    def _write_record(self, record: dict):
        # print(record)
        with _lock:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write("=" * 50 + "\n")
                f.write(f"Call N {record['call_number']} at {record['timestamp']}\n")
                f.write("=" * 50 + "\n")
                f.write(f"{record['prompt']}\n\n")
                f.write("Model response:\n")
                f.write(f"{record.get('response',{}).get('content','')}")
                f.write(f"{record.get('exception', '')}")
                f.write("\n\n\n\n")


def instantiate_llm(config: Config) -> ChatGoogleGenerativeAI:
    model = LoggedChatModel(
        ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.model_settings.temperature,
            project=config.gcp_project,
        )
    )

    return model
