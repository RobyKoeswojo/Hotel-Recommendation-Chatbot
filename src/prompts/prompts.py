from langchain_core.prompts import PromptTemplate
from abc import abstractmethod


class BasePrompt:
    """
    The base class for the prompt template.
    """

    def __init__(self):
        """
        Initialize the prompt class.
        """
        self.head_template = """
        You are an AI customer service that gives recommendation about hotels to users.
        Answer the following questions as best you can.
        """

        self.chat_history_template = """
        Previous conversation history:
        {chat_history}
        """

        self.final_template = ""

    def __str__(self) -> str:
        """
        Print out the final template.
        """
        return self.final_template

    @abstractmethod
    def get(self) -> PromptTemplate:
        """
        Retrieve the prompt template.

        Returns
        -------
        PromptTemplate
            The prompt template.
        """
        pass


class ReactPrompt(BasePrompt):
    def __init__(self, conversation_history: bool):
        """
        Initialize the prompt template with ReAct style.

        Parameters
        ----------
        conversation_history: bool
            If True, the prompt template will add chat history as context.
        """
        super().__init__()
        self.conversation_history = conversation_history
        self.body_template = """
        You have access to the following tools:

        {tools}

        Use the following format:
    
        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question
    
        Begin!
        
        """

        self.end_template = """
        Question: {input}
        Thought: {agent_scratchpad}
        """

        if self.conversation_history:
            chat_history_template = """Previous conversation history:
        {chat_history}
            """
            self.final_template = self.head_template + self.body_template + \
                                  chat_history_template + self.end_template
        else:
            self.final_template = self.head_template + self.body_template + \
                                  self.end_template

    def get(self) -> PromptTemplate:
        input_variables = ['agent_scratchpad',
                           'input',
                           'chat_history',
                           'tool_names',
                           'tools']
        if self.conversation_history:
            input_variables.append('chat_history')
        return PromptTemplate(
            input_variables=input_variables,
            template=self.final_template
        )


class RAGPrompt(BasePrompt):
    def __init__(self):
        """
        Initialize the prompt for RAG only purpose.
        """
        super().__init__()
        self.body_template = """
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say you don't know.
        Use three sentences maximum and keep the answer concise.
        
        Question: {input}
        Context: {context}
        Answer:
        """
        self.final_template = self.head_template + self.body_template

    def get(self) -> PromptTemplate:
        input_variables = ['input', 'context']
        return PromptTemplate(
            input_variables=input_variables,
            template=self.final_template
        )
