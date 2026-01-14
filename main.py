from langchain.messages import HumanMessage
from src.agent import compile, call

if __name__ == "__main__":
    agent = compile()
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="fetch me the list of addresses of the facilities our company owns"
                )
            ]
        }
    )

    print(type(result))
    print(result)
    print()

    print(len(result["messages"]))
    print()

    print(result["messages"][-1].content[0]["text"])
    print()
