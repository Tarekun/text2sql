from langchain.messages import HumanMessage
from src.agent.graph import compile, call
from src.config import read_config
from src.db import gcp_pull_metadata, get_table_metadata
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument(
        "--config", type=str, default="config.yml", help="Path to the config file"
    )
    args = parser.parse_args()

    config = read_config(args.config)
    agent = compile(config)

    # messages = agent.invoke(
    #     {
    #         "messages": [
    #             HumanMessage(
    #                 content="i need to know TOTAL average daily income,in year 2025, grouped by STRUTTURA. underlying db is BigQuery so never use function STRFTIME as its not supported"
    #             )
    #         ]
    #     }
    # )
    # print()
    # print("risultato")
    # for m in messages["messages"]:
    #     print(m.__class__)
    #     print(m.content)
    #     print()

    while True:
        question = input("> ")
        if question == "/quit":
            break

        answer = call(agent, question)
        print(answer)
    print("Bye!")
