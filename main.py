from langchain.messages import HumanMessage
from src.agent import compile, call
from src.db import gcp_pull_metadata, get_table_metadata

# gcp_pull_metadata("soges-group-data-platform", datasets=["gold"])
# print(get_table_metadata())

question = "voglio sapere l'incasso TOTALE giornaliero medio, nell'anno 2025, diviso per struttura"
if __name__ == "__main__":
    agent = compile()
    result = call(agent, question)

    print(f"Question: {question}")
    print(result)
