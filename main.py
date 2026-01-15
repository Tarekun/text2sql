from langchain.messages import HumanMessage
from src.agent.graph import compile, call
from src.config import read_config
from src.db import gcp_pull_metadata, get_table_metadata
from src.utils import print_graph
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument(
        "--config", type=str, default="config.yml", help="Path to the config file"
    )
    args = parser.parse_args()

    config = read_config(args.config)
    agent = compile(config)
    print_graph(agent)

    # messages = agent.invoke(
    #     {
    #         "messages": [
    #             HumanMessage(
    #                 content="find me all the addresses (INDIRIZZI) of the buildings (STRUTTURA) tracked on the db"
    #             )
    #         ]
    #     }
    # )
    # print()
    # print("risultato")
    # # for m in messages["messages"]:
    # m = messages["messages"][-1]
    # print(m.__class__)
    # print(m)
    # print()

    while True:
        question = input("> ")
        if question == "/quit":
            break

        answer = call(agent, question)
        print(answer)
    print("Bye!")
