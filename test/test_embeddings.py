import unittest, os
from src.embeddings import Embeddings, EmbeddingType
from dotenv import load_dotenv

ENV_DIR = os.path.join(os.path.dirname(os.getcwd()), ".env")
load_dotenv(ENV_DIR)


class TestEmbeddings(unittest.TestCase):
    def test_get(self):
        embedding_model = Embeddings.get(EmbeddingType.SENTENCE_TRANSFORMER)
        self.assertIsNotNone(embedding_model, ValueError)
        print(embedding_model)


if __name__ == "__main__":
    unittest.main()
