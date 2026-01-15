from langchain.messages import HumanMessage
from src.agent.graph import compile, call
from src.config import read_config
from src.db import gcp_pull_metadata, get_table_metadata
import argparse


# gcp_pull_metadata("soges-group-data-platform", datasets=["gold"])
# print(get_table_metadata())

# question = "voglio sapere l'incasso TOTALE giornaliero medio, nell'anno 2025, diviso per struttura. il db sottostante è bigquery quindi non usare mai la funzione STRFTIME che non è supportata"
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument(
        "--config", type=str, default="config.yml", help="Path to the config file"
    )
    args = parser.parse_args()

    config = read_config(args.config)
    agent = compile(config)

    while True:
        question = input("> ")
        if question == "/quit":
            break

        answer = call(agent, question)
        print(answer)
    print("Bye!")
