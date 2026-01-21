from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langchain.messages import (
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
NODE_EXECUTE_NAME = "execute_sql"
NODE_ANSWER_NAME = "answer"
NODE_TOOLS_NAME = "tools"
NODE_POST_TOOL_NAME = "tool_state_mngt"
NODE_PYTHON_GENERATION_NAME = "python_generation"
NODE_PYTHON_POST_TOOL_NAME = "python_tool_state_mngt"
NODE_PYTHON_INTERPRETER_NAME = "python_interpreter"
NODE_SUFFEVAL_NAME = "context_eval"
llm_control_nodes = [
    NODE_GENERATE_NAME,
    NODE_PYTHON_GENERATION_NAME,
]
llm_nodes = [
    NODE_ANSWER_NAME,
    NODE_SUFFEVAL_NAME,
]
tool_nodes = [NODE_TOOLS_NAME, NODE_PYTHON_INTERPRETER_NAME]

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
        agent_builder.add_node(NODE_TOOLS_NAME, ToolNode([execute_sql, fetch_metadata]))
        agent_builder.add_node(
            NODE_PYTHON_INTERPRETER_NAME, ToolNode([python_interpreter])
        )
        agent_builder.add_node(NODE_POST_TOOL_NAME, self._node_post_data_tool)
        agent_builder.add_node(NODE_PYTHON_POST_TOOL_NAME, self._node_post_python_tool)
        agent_builder.add_node(NODE_SUFFEVAL_NAME, self._node_sufficiency_evaluation)
        agent_builder.add_node(
            NODE_PYTHON_GENERATION_NAME, self._node_python_execution_sql
        )
        agent_builder.add_node(NODE_ANSWER_NAME, self._node_final_answer)
        # Add edges to connect nodes
        agent_builder.add_edge(START, NODE_GENERATE_NAME)
        agent_builder.add_conditional_edges(
            NODE_GENERATE_NAME,
            self._edge_skip_execution,
            [NODE_TOOLS_NAME, NODE_PYTHON_GENERATION_NAME],
        )
        agent_builder.add_edge(NODE_TOOLS_NAME, NODE_POST_TOOL_NAME)
        agent_builder.add_edge(NODE_POST_TOOL_NAME, NODE_SUFFEVAL_NAME)
        agent_builder.add_conditional_edges(
            NODE_SUFFEVAL_NAME,
            self._edge_sufficiency_evaluation,
            [NODE_GENERATE_NAME, NODE_PYTHON_GENERATION_NAME],
        )
        agent_builder.add_edge(
            NODE_PYTHON_GENERATION_NAME, NODE_PYTHON_INTERPRETER_NAME
        )
        agent_builder.add_edge(NODE_PYTHON_INTERPRETER_NAME, NODE_PYTHON_POST_TOOL_NAME)
        agent_builder.add_conditional_edges(
            NODE_PYTHON_POST_TOOL_NAME,
            self._edge_python_successful_execution,
            [NODE_PYTHON_GENERATION_NAME, NODE_ANSWER_NAME],
        )
        agent_builder.add_edge(NODE_ANSWER_NAME, END)

        self.graph: CompiledStateGraph = agent_builder.compile()

    def invoke(self, message: str):
        messages = self.graph.invoke({"messages": [HumanMessage(content=message)]})
        return content_as_string(messages["messages"][-1])

    def _node_generate_sql(self, state: MessagesState):
        logger.debug("node: main control node")

        user_query = get_user_question(state)
        metadata = state.get("metadata", "No metadata fetched yet")
        data = state.get("fetched_data", "No rows fetched yet")
        system_prompt = self.local_prompts.sql_generation.format(
            metadata=metadata,
            data=data,
            db_kind="BigQuery",
        )
        llm = self.llm.bind_tools([execute_sql, fetch_metadata])
        return retryable_generation(
            state,
            llm_with_tools=llm,
            system_prompt=system_prompt,
            max_retries=self.max_retries,
            user_query=user_query,
            retry_prompt="The above SQL query failed. Please analyze the error and generate a corrected query.",
            detect_error=did_last_sql_run_fail,
        )

    def _node_python_execution_sql(self, state: MessagesState):
        logger.debug("node: python execution node")
        llm = self.llm.bind_tools([python_interpreter])
        user_query = get_user_question(state)
        data = state.get("fetched_data", "No data fetched")
        python_output = state.get("python_output", "No previous python executions")
        system_prompt = self.local_prompts.python_opt_generation.format(
            data=data,
            python_output=python_output,
        )
        return retryable_generation(
            state,
            llm_with_tools=llm,
            system_prompt=system_prompt,
            max_retries=self.max_retries,
            user_query=user_query,
            retry_prompt="The previous python script failed. Please analyze the error and generate a corrected program.",
            detect_error=did_last_python_run_fail,
        )

    def _node_post_data_tool(self, state: MessagesState):
        logger.debug("node: post tool state management")
        retry = 0
        if did_last_sql_run_fail(state):
            logger.warning("SQL execution failed, retrying")
            retry = state.get("retry_count", 0) + 1

        return {
            "retry_count": retry,
            "metadata": get_fetched_metadata(state),
            "fetched_data": get_fetched_data(state),
        }

    def _node_post_python_tool(self, state: MessagesState):
        logger.debug("node: post tool state management")
        retry = 0
        if did_last_python_run_fail(state):
            logger.warning("Python execution failed, retrying")
            retry = state.get("retry_count", 0) + 1

        return {
            "retry_count": retry,
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

    def _node_sufficiency_evaluation(self, state: MessagesState):
        logger.debug("node: context evaluation")
        user_query = get_user_question(state)
        metadata = state.get("metadata", "No metadata fetched yet")
        data = state.get("fetched_data", "No rows fetched yet")
        python_output = state.get("python_output", "No previous python executions")
        system_prompt = self.local_prompts.evaluate_context.format(
            metadata=metadata,
            data=data,
            user_query=user_query,
            python_output=python_output,
        )

        response = self.llm.invoke(
            [
                # gemini api always wants a system message AND a human message
                SystemMessage(
                    content="You are a strict evaluator. Respond only with DATA IS EXAUSTIVE or MISSING DATA."
                ),
                HumanMessage(content=system_prompt),
            ]
        )
        response = content_as_string(response)
        # TODO "DATA IS EXAUSTIVE" is hard coded here and should be fixed somehow
        return {"sufficient_context": "DATA IS EXAUSTIVE" in response.upper()}

    def _edge_skip_execution(self, state: MessagesState) -> str:
        """Routes to tool sql execution or final answer generation depending on
        if the model produced a sql query tool call in the previous message"""
        last_message = state["messages"][-1]
        logger.debug("edge: skip tools to final execution")

        if (
            hasattr(last_message, "tool_calls")
            and last_message.tool_calls  # type:ignore
        ):
            logger.debug("going to tool calls")
            return NODE_TOOLS_NAME
        else:
            logger.debug("going to final generation")
            return NODE_ANSWER_NAME

    def _edge_sufficiency_evaluation(self, state: MessagesState) -> str:
        logger.debug("edge: sufficient context branching")
        if state["sufficient_context"]:
            logger.debug("proceeding to answer")
            return NODE_PYTHON_GENERATION_NAME
        else:
            logger.debug("looping again")
            return NODE_GENERATE_NAME

    def _edge_python_successful_execution(self, state: MessagesState) -> str:
        logger.debug("edge: python execution result check")
        if did_last_python_run_fail(state) and state["retry_count"] < self.max_retries:
            return NODE_PYTHON_GENERATION_NAME
        else:
            return NODE_ANSWER_NAME


def retryable_generation(
    state: MessagesState,
    llm_with_tools,
    system_prompt: str,
    user_query: str,
    retry_prompt: str,
    max_retries: int,
    detect_error: Callable[[MessagesState], bool],
):
    """Reusable function that implements a node with a retryable tool call."""

    if state.get("retry_count", 0) > max_retries:
        logger.error("Tool usage failed too many times. Skipping")
        return {
            "messages": [
                HumanMessage(content="Tool usage failed too many times. Skipping")
            ],
        }

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query),
    ]
    if detect_error(state):
        # add to context the failed generated code
        messages.append(state["messages"][-2])
        # add to context the error message
        tool_error_msg = state["messages"][-1]
        messages.append(tool_error_msg)
        messages.append(HumanMessage(content=retry_prompt))

    response = llm_with_tools.invoke(messages)
    return {
        "messages": [response],
    }
