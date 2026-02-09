from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langchain.messages import (
    ToolMessage,
    SystemMessage,
    HumanMessage,
)
from langgraph.prebuilt import ToolNode
from typing import Callable
from src.agent.llm_backend import instantiate_llm
from src.agent.state import *
from src.agent.tools import *
from src.config import Config
from src.prompts import prompts
from src.utils import get_user_question, content_as_string
from src.logger import logger


NODE_GENERATE_NAME = "data_fetching"
NODE_ANSWER_NAME = "answer"
NODE_TOOLS_NAME = "tools"
NODE_POST_TOOL_NAME = "tool_state_mngt"
llm_control_nodes = [
    NODE_GENERATE_NAME,
]
llm_nodes = [
    NODE_ANSWER_NAME,
]
tool_nodes = [
    NODE_TOOLS_NAME,
]

EXECUTION_ERROR_PREFIX = "SQL execution error:"


class Text2SqlAgent:
    def __init__(self, config: Config):
        self.max_retries = config.max_retries
        self.local_prompts = prompts[config.language]
        self.llm = instantiate_llm(config)

        # Build workflow
        agent_builder = StateGraph(state_schema=MessagesState)

        # Add nodes
        agent_builder.add_node(NODE_GENERATE_NAME, self._node_generate_sql)
        agent_builder.add_node(
            NODE_TOOLS_NAME,
            ToolNode(all_tools),
        )
        agent_builder.add_node(NODE_POST_TOOL_NAME, self._node_post_data_tool)
        agent_builder.add_node(NODE_ANSWER_NAME, self._node_final_answer)

        # Add edges to connect nodes
        agent_builder.add_edge(START, NODE_GENERATE_NAME)
        agent_builder.add_conditional_edges(
            NODE_GENERATE_NAME,
            self._edge_skip_execution,
            [NODE_TOOLS_NAME, NODE_ANSWER_NAME],
        )
        agent_builder.add_edge(NODE_TOOLS_NAME, NODE_POST_TOOL_NAME)
        agent_builder.add_edge(NODE_POST_TOOL_NAME, NODE_GENERATE_NAME)
        agent_builder.add_edge(NODE_ANSWER_NAME, END)

        self.graph: CompiledStateGraph = agent_builder.compile()

    def invoke(self, message: str):
        messages = self.graph.invoke({"messages": [HumanMessage(content=message)]})
        return content_as_string(messages["messages"][-1])

    def _node_generate_sql(self, state: MessagesState):
        logger.debug("node: main control node")
        if state.get("retry_count", 0) > self.max_retries:
            logger.error("Tool usage failed too many times. Skipping")
            return {
                "messages": [
                    ToolMessage(content="Tool usage failed too many times. Skipping")
                ],
            }

        # base prompt construction
        llm = self.llm.bind_tools(all_tools)
        user_query = get_user_question(state)
        metadata = state.get("metadata", "No metadata fetched yet")
        data = state.get("fetched_data", "No rows fetched yet")
        topk_queries = state.get("topk_queries", "No query lookup performed yet")
        python_output = state.get("python_output", "No previous python executions")
        system_prompt = self.local_prompts.sql_generation.format(
            metadata=metadata,
            data=data,
            db_kind="BigQuery",
            topk_queries=topk_queries,
            python_output=python_output,
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_query),
        ]

        # previous error handling
        if did_last_sql_run_fail(state):
            # add to context the failed generated code
            messages.append(state["messages"][-2])
            # add to context the error message
            tool_error_msg = state["messages"][-1]
            messages.append(tool_error_msg)
            messages.append(
                HumanMessage(
                    content="The above SQL query failed. Please analyze the error and generate a corrected query."
                )
            )
        if did_last_python_run_fail(state):
            # add to context the failed generated code
            messages.append(state["messages"][-2])
            # add to context the error message
            tool_error_msg = state["messages"][-1]
            messages.append(tool_error_msg)
            messages.append(
                HumanMessage(
                    content="The previous python script failed. Please analyze the error and generate a corrected program."
                )
            )

        # llm call
        response = llm.invoke(messages)
        return {
            "messages": [response],
        }

    def _node_post_data_tool(self, state: MessagesState):
        logger.debug("node: post tool state management")
        retry = 0
        if did_last_sql_run_fail(state):
            logger.warning("SQL execution failed, retrying")
            retry = state.get("retry_count", 0) + 1
        if did_last_python_run_fail(state):
            logger.warning("Python execution failed, retrying")
            retry = state.get("retry_count", 0) + 1

        return {
            "retry_count": retry,
            "metadata": get_fetched_metadata(state),
            "fetched_data": get_fetched_data(state),
            "topk_queries": get_topk_queries(state),
            "python_output": get_python_output(state),
        }

    def _node_final_answer(self, state: MessagesState):
        logger.debug("node: final answer")
        user_query = get_user_question(state)
        metadata = state.get("metadata", "No metadata fetched yet")
        sql_result = state.get("fetched_data", "No rows fetched yet")
        python_output = state.get("python_output", "No previous python executions")

        if sql_result is None:
            logger.debug("Final answer has no SQL data available")
        if metadata is None:
            logger.debug("Final answer has not metadata available")
        system_prompt = self.local_prompts.final_answer.format(
            data=sql_result,
            metadata=metadata,
            python_output=python_output,
        )
        messages = [("system", system_prompt), ("human", user_query)]

        response = self.llm.invoke(messages)
        return {"messages": [response]}

    def _edge_skip_execution(self, state: MessagesState) -> str:
        """Routes to tool sql execution or final answer generation depending on
        if the model produced a sql query tool call in the previous message"""
        last_message = state["messages"][-1]

        if (
            hasattr(last_message, "tool_calls")
            and last_message.tool_calls  # type:ignore
        ):
            logger.debug("edge: going to tool calls")
            return NODE_TOOLS_NAME
        else:
            logger.debug("edge: going to final generation")
            return NODE_ANSWER_NAME
