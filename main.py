from langchain.messages import HumanMessage
from src.agent.graph import Text2SqlAgent, llm_nodes, tool_nodes, llm_control_nodes
from src.cache.metadata import get_table_metadata
from src.config import read_config, get_args_parser
from src.logger import configure_logger, logger
from src.utils import print_graph
import argparse


if __name__ == "__main__":
    args = get_args_parser()
    config = read_config()
    configure_logger(config)
    # has to be after the configure_logger call
    logger.debug(f"Loaded config: {config}")

    agent = Text2SqlAgent(config)
    print_graph(
        agent.graph,
        llm_nodes=llm_nodes,
        llm_control_nodes=llm_control_nodes,
        tool_nodes=tool_nodes,
    )

    if args.question:
        answer = agent.invoke(args.question)
        print(answer)
    else:
        while True:
            question = input("> ")
            if question == "/quit":
                break

            answer = agent.invoke(question)
            print(answer)
        print("Bye!")
