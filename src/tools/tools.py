from langchain.tools.retriever import create_retriever_tool
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import Tool
from abc import abstractmethod


class Tools:
    @classmethod
    @abstractmethod
    def get(cls):
        return


class RetrieverTool(Tool):
    @classmethod
    def get(cls, retriever):
        return create_retriever_tool(
            retriever=retriever,
            name="retriever-tool",
            description="Search related documents to answer questions"
        )


class OnlineSearchTool(Tool):
    @classmethod
    def get(cls):
        search = SerpAPIWrapper()
        return Tool(
            name="online-search-tool",
            description="Online search to answer questions",
            func=search.run
        )
