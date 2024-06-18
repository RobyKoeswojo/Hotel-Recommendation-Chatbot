from langchain.agents import create_react_agent, AgentExecutor


class Agents:
    @classmethod
    def get(cls, llm, tools, prompt, react, conversation_history, verbose):
        if react:
            agent = create_react_agent(llm, tools, prompt)
            return AgentExecutor(
                agent=agent,
                tools=tools,
                return_intermediate_steps=True,
                handle_parsing_errors=True,
                verbose=verbose
            )
        else:
            pass

