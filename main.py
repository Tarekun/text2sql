from langchain.messages import HumanMessage
from src.agent.graph import compile, call
from src.db import gcp_pull_metadata, get_table_metadata
import argparse


# gcp_pull_metadata("soges-group-data-platform", datasets=["gold"])
# print(get_table_metadata())

question = "voglio sapere l'incasso TOTALE giornaliero medio, nell'anno 2025, diviso per struttura. il db sottostante è bigquery quindi non usare mai la funzione STRFTIME che non è supportata"
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument(
        "--config", type=str, default="config.yml", help="Path to the config file"
    )
    args = parser.parse_args()

    agent = compile()
    # result = call(agent, question)
    messages = agent.invoke({"messages": [HumanMessage(content=question)]})
    print(messages)

    # print(f"Question: {question}")
    # print(result)
