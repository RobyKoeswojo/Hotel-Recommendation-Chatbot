from langchain_community.embeddings import (HuggingFaceEmbeddings,
                                            OpenAIEmbeddings)
from .embedding_type import EmbeddingType
from typing import Union

REGISTRY_EMBEDDING = {
    EmbeddingType.SENTENCE_TRANSFORMER: "all-MiniLM-L6-v2",
    EmbeddingType.OPENAI_EMBEDDING_SMALL: "text-embedding-3-small"
}


class Embeddings:
    """
    A class to get the embedding model.
    """
    @classmethod
    def get(cls, embedding_name: EmbeddingType) -> Union[HuggingFaceEmbeddings,
                                                         OpenAIEmbeddings]:
        """
        Retrieve the desired embedding model.

        Parameters
        ----------
        embedding_name: EmbeddingType
            The name of desired embedding model.

        Returns
        -------
        Union[HuggingFaceEmbeddings, OpenAIEmbeddings]
            The desired embedding model.

        Raises
        ------
        NotImplementedError
            If the desired embedding model has not been implemented.
        """
        if embedding_name == EmbeddingType.SENTENCE_TRANSFORMER:
            print(f"[INFO] Using {EmbeddingType.SENTENCE_TRANSFORMER}")
            return HuggingFaceEmbeddings(
                model_name=REGISTRY_EMBEDDING[embedding_name]
            )
        elif embedding_name == EmbeddingType.OPENAI_EMBEDDING_SMALL:
            print(f"[INFO] Using {EmbeddingType.OPENAI_EMBEDDING_SMALL}")
            return OpenAIEmbeddings(
                model=REGISTRY_EMBEDDING[embedding_name])
        else:
            raise NotImplementedError("Other embedding models have not been"
                                      "implemented.")

