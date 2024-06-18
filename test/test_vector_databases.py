import unittest
from src.vector_databases import ChromaDB
from src.embeddings import Embeddings, EmbeddingType


class TestChromaDB(unittest.TestCase):
    def test_get(self):
        embedding_model = Embeddings.get(EmbeddingType.SENTENCE_TRANSFORMER)
        vector_db = ChromaDB.get(embedding_model, "United Kingdom")
        self.assertIsNotNone((vector_db, ValueError))
        sim_docs = vector_db.similarity_search("London", k=3)
        self.assertEqual(len(sim_docs), 3)
        print(sim_docs)


if __name__ == "__main__":
    unittest.main()
