import os
from data_preparation import CSVData
from models import Models
from embeddings import Embeddings
from vector_databases import ChromaDB
from tools import RetrieverTool, OnlineSearchTool, ToolType
from prompts import ReactPrompt, RAGPrompt
from agents import Agents
from dotenv import load_dotenv
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory


ENV_DIR = os.path.join(os.path.dirname(os.getcwd()), ".env")
load_dotenv(ENV_DIR)


def run(model_name="chat-gpt-3.5", embedding_name="sentence-transformer",
        country="United Kingdom",
        tools_list=["retriever-tool",
                    # "online-search-tool"
                    ],
        react=True,
        verbose = True,
        conversation_history = True
):

    CSVData("Hotel_Reviews").create_processed_data(country)

    llm = Models.get(model_name)
    embedding_model = Embeddings.get(embedding_name)

    tools = []
    if ToolType.RETRIEVER in tools_list:
        print("Adding retriever tool.")
        vector_db = ChromaDB.get(embedding_model, country)
        retriever = vector_db.as_retriever(k=3)
        tools.append(RetrieverTool.get(retriever))
    if ToolType.ONLINE_SEARCH in tools_list:
        print("Adding online search tool")
        tools.append(OnlineSearchTool.get())

    prompt = ""
    if react:
        prompt = ReactPrompt(
            conversation_history=conversation_history).get()
    agent = Agents.get(llm, tools, prompt,
                       react, conversation_history, verbose)
    if conversation_history:
        store = {}

        def get_session_history(session_id: str):
            if session_id not in store:
                store[session_id] = ChatMessageHistory()
            return store[session_id]

        agent_executor = RunnableWithMessageHistory(
            agent,
            get_session_history,
            input_messages_key='input',
            history_messages_key='chat_history'
        )
    else:
        agent_executor = agent

    while True:
        print("\n=============================================================")
        print("\n=============================================================")
        question = input("\nQuestion:")
        if question.lower() == "stop": break
        full_question = f"Use the 'retriever_tool' first to answer: {question}. If cannot get the answer, use other tool"

        response = agent_executor.invoke(
            {"input": full_question},
            config={"configurable": {"session_id": "123"}},
        )
        print(response['output'])


if __name__ == "__main__":
    run()
