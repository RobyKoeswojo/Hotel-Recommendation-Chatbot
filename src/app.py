from data_preparation import CSVData
from models import Models, ModelType
from embeddings import Embeddings, EmbeddingType
from vector_databases import ChromaDB
from tools import RetrieverTool, OnlineSearchTool, ToolType
from prompts import ReactPrompt
from agents import Agents
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory

import os
from dotenv import load_dotenv
import streamlit as st

ENV_DIR = os.path.join(os.path.dirname(os.getcwd()), ".env")
load_dotenv(ENV_DIR)


def run(model_name, embedding_name, country="United Kingdom",
        online_search=False):
    CSVData("Hotel_Reviews").create_processed_data(country)
    llm = Models.get(model_name=model_name)
    embedding_model = Embeddings.get(embedding_name=embedding_name)

    vector_db = ChromaDB.get(
        embedding_model=embedding_model,
        country=country)
    retriever = vector_db.as_retriever(search_kwargs={'k': 3})
    tools = [RetrieverTool.get(retriever)]
    if online_search:
        tools.append(OnlineSearchTool.get())

    # use default setting: react prompt
    prompt = ReactPrompt(conversation_history=True).get()
    agent = Agents.get(llm=llm,
                       tools=tools,
                       prompt=prompt,
                       react=True,
                       verbose=True)
    # use default setting: chat memory
    store = {}

    def get_session_history(session_id: str):
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    # TODO: fix "treating as root run..'
    agent_executor = RunnableWithMessageHistory(
        agent,
        get_session_history,
        input_messages_key='input',
        history_messages_key='chat_history'
    )

    st.text("------------- Chatting -------------")
    question = st.text_input("Your question: ")
    full_question = f"Use the 'retriever_tool' first to answer: {question}. " \
                    f"If cannot get the answer, use other tool"
    submit = st.button("Ask!")

    # only response when all components are ready
    if agent_executor and full_question:
        print(f"[INFO] Agent is ready!")
        if submit:
            print(f"[INFO] Question: {full_question}")
            response = agent_executor.invoke(
                {"input": full_question},
                config={"configurable": {"session_id": "test-session"}},
                    )
            st.text(response['output'])


if __name__ == "__main__":
    st.title("StayChat: A Hotel Recommendation Chatbot")

    # User input parameters
    model_options = ["ChatGPT 3.5", "Phi3 4K"]
    REGISTRY_MODEL = {
        model_options[0]: ModelType.CHATGPTSTANDARD,
        model_options[1]: ModelType.PHITHREE4k
    }
    selected_model = st.selectbox("Select model: ", model_options)
    selected_model = REGISTRY_MODEL[selected_model]

    embedding_options = ["Sentence Transformer", "OpenAI Embedding Small"]
    REGISTRY_EMBEDDING = {
        embedding_options[0]: EmbeddingType.SENTENCE_TRANSFORMER,
        embedding_options[1]: EmbeddingType.OPENAI_EMBEDDING_SMALL
    }
    selected_embedding = st.selectbox("Select embedding: ", embedding_options)
    selected_embedding = REGISTRY_EMBEDDING[selected_embedding]

    selected_country = st.selectbox("Select country: ", ["United Kingdom"])

    REGISTRY_SEARCH = {"No": False, "Yes": True}
    use_online_search = st.selectbox("Use online search? ", ["No", "Yes"])
    use_online_search = REGISTRY_SEARCH[use_online_search]

    run(model_name=selected_model,
        embedding_name=selected_embedding,
        online_search=use_online_search)

