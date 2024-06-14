from langchain_community.embeddings import (HuggingFaceEmbeddings,
                                            OpenAIEmbeddings)
from .embedding_type import EmbeddingType

REGISTRY_EMBEDDING = {
    EmbeddingType.SENTENCE_TRANSFORMER: "all-MiniLM-L6-v2",
    EmbeddingType.OPENAI_EMBEDDING_SMALL: "text-embedding-3-small"
}


class Embeddings:
    @classmethod
    def get(cls, embedding_name):
        if embedding_name == EmbeddingType.SENTENCE_TRANSFORMER:
            return HuggingFaceEmbeddings(
                model_name=REGISTRY_EMBEDDING[embedding_name]
            )
        else:
            return OpenAIEmbeddings(
                model=REGISTRY_EMBEDDING[embedding_name])
