import unittest, os
from src.models import Models, ModelType
from src.tools import RetrieverTool, ToolType
from src.vector_databases import ChromaDB
from src.embeddings import Embeddings, EmbeddingType
from src.prompts import ReactPrompt
from src.agents import Agents
from dotenv import load_dotenv

ENV_DIR = os.path.join(os.path.dirname(os.getcwd()), ".env")
load_dotenv(ENV_DIR)


class TestAgents(unittest.TestCase):
    def test_get(self):
        llm = Models.get(ModelType.CHATGPTSTANDARD)
        embedding_model = Embeddings.get(EmbeddingType.SENTENCE_TRANSFORMER)
        vector_db = ChromaDB.get(embedding_model, "United Kingdom")
        retriever = vector_db.as_retriever(search_kwargs={'k': 3})
        prompt = ReactPrompt(conversation_history=True).get()
        tools = [RetrieverTool.get(retriever)]
        agent = Agents.get(llm, tools, prompt,
                           react=True, verbose=False)

        self.assertIsNotNone(agent, ValueError)


if __name__ == "__main__":
    unittest.main()
