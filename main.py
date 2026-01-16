from langchain.messages import HumanMessage
from src.agent.graph import Text2SqlAgent
from src.config import read_config
from src.db import gcp_pull_metadata, get_table_metadata
from src.logger import configure_logger
from src.utils import print_graph
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument(
        "--config", type=str, default="config.yml", help="Path to the config file"
    )
    args = parser.parse_args()

    config = read_config(args.config)
    configure_logger(config)
    agent = Text2SqlAgent(config)
    print_graph(agent.graph)

    # messages = agent.graph.invoke(
    #     {
    #         "messages": [
    #             HumanMessage(content="what is the table with the most byte usage")
    #         ]
    #     }
    # )
    # print()
    # print("risultato")
    # for m in messages["messages"]:
    #     # m = messages["messages"][-1]
    #     print(m.__class__)
    #     print(m)
    #     print()

    while True:
        question = input("> ")
        if question == "/quit":
            break

        answer = agent.invoke(question)
        print(answer)
    print("Bye!")
