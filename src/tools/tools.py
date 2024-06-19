from langchain.tools.retriever import create_retriever_tool
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import Tool
from langchain_core.vectorstores import VectorStoreRetriever
from abc import abstractmethod


class Tools:
    """
    The base class for tools.
    """
    @classmethod
    @abstractmethod
    def get(cls) -> Tool:
        """
        Retrieve the desired tool.

        Returns
        -------
        Tool
            The desired tool.
        """
        pass


class RetrieverTool(Tool):
    """
    The class to create retriever tool.
    """
    @classmethod
    def get(cls, retriever: VectorStoreRetriever) -> Tool:
        """
        Retrieve the retriever tool

        Parameters
        ----------
        retriever: VectorStoreRetriever
            The retriever of the vector store.
        """
        print(f"[INFO] Using Retriever Tool")
        return create_retriever_tool(
            retriever=retriever,
            name="retriever-tool",
            description="Search related documents to answer questions"
        )


class OnlineSearchTool(Tool):
    """
    The class to create the tool to run online search.
    """
    @classmethod
    def get(cls) -> Tool:
        print(f"[INFO] Using Online Search Tool")
        search = SerpAPIWrapper()
        return Tool(
            name="online-search-tool",
            description="Online search to answer questions",
            func=search.run
        )
