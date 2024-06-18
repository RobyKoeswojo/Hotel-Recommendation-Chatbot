from enum import Enum


class EmbeddingType(str, Enum):
    SENTENCE_TRANSFORMER = "sentence-transformer"
    OPENAI_EMBEDDING_SMALL = "openai-embedding-small"
