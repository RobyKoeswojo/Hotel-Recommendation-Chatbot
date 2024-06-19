import unittest, os
from src.tools import RetrieverTool, OnlineSearchTool
from src.vector_databases import ChromaDB
from src.embeddings import Embeddings, EmbeddingType

from dotenv import load_dotenv
ENV_DIR = os.path.join(os.path.dirname(os.getcwd()), ".env")
load_dotenv(ENV_DIR)


class TestTools(unittest.TestCase):
    def test_get(self):
        embedding_model = Embeddings.get(EmbeddingType.SENTENCE_TRANSFORMER)
        vector_db = ChromaDB.get(embedding_model, "United Kingdom")
        retriever = vector_db.as_retriever(search_kwargs={'k': 3})
        retriever_tool = RetrieverTool.get(retriever)
        online_search_tool = OnlineSearchTool.get()

        self.assertIsNotNone(retriever_tool, ValueError)
        self.assertIsNotNone(online_search_tool, ValueError)


if __name__ == "__main__":
    unittest.main()
