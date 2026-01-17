from dataclasses import dataclass


@dataclass(frozen=True)  # frozen=True makes it immutable (good for config)
class Prompts:
    sql_generation: str
    final_answer: str
    evaluate_context: str
    python_execution: str
