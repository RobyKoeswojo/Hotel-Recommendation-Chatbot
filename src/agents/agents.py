from langchain.agents import create_react_agent, AgentExecutor, Tool
from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from typing import Union, List


class Agents:
    """
    The base class to create an agent.
    """
    @classmethod
    def get(cls,
            llm: Union[ChatOpenAI, HuggingFaceEndpoint],
            tools: List[Tool],
            prompt: PromptTemplate,
            react: bool,
            verbose: bool = False) -> AgentExecutor:
        """
        Get an agent based on the provided parameters.

        Parameters
        ----------
        llm: Union[ChatOpenAI, HuggingFaceEndpoint]
            The LLM model.
        tools: List[Tool]
            The tools for the agent.
        prompt: PromptTemplate
            The prompt template for the agent.
        react: bool
            If True, creating react agent.
        verbose: bool, optional
            If True, print detailed progress and debug information during run.
            Defaults to False.

        Returns
        -------
        AgentExecutor
            The agent executor.

        Raises
        ------
        NotImplementedError
            If the desired agent type has not been implemented.
        """
        if react:
            print("[INFO] Creating React Agent.")
            agent = create_react_agent(llm, tools, prompt)
            return AgentExecutor(
                agent=agent,
                tools=tools,
                return_intermediate_steps=True,
                handle_parsing_errors=True,
                verbose=verbose
            )
        else:
            raise NotImplementedError(
                "Other prompt style implementation is needed."
            )  # need to implement if not using react prompt

