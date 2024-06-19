import os
from langchain_community.embeddings import (HuggingFaceEmbeddings,
                                            OpenAIEmbeddings)
from langchain_community.vectorstores import Chroma
from langchain.document_loaders.csv_loader import CSVLoader
from abc import abstractmethod
from typing import Union

DATA_DIR = os.path.join(os.path.dirname(os.getcwd()), 'data')


class VectorDatabase:
    """
    The base class for vector database that stores the documents for retriever
    tool.
    """
    @classmethod
    @abstractmethod
    def get(cls,
            embedding_model: Union[HuggingFaceEmbeddings, OpenAIEmbeddings],
            country: str):
        """
        Retrieve the vector database.
        """
        pass

    @classmethod
    @abstractmethod
    def _create_db(cls,
                   embedding_model: Union[HuggingFaceEmbeddings,
                                          OpenAIEmbeddings],
                   country: str):
        pass

    @classmethod
    @abstractmethod
    def _load_db(cls, embedding_model: Union[HuggingFaceEmbeddings,
                                             OpenAIEmbeddings]):
        pass

    @staticmethod
    def _check_path_exist(path) -> bool:
        """
        Check whether the path already exists and if it already has content.

        Parameters
        ----------
        path: str
            The path (directory) to be checked.

        Returns
        -------
        bool
            If True, the path already exists, and it already has content inside.
        """
        if os.path.exists(path):
            return len(os.listdir(path)) > 0
        else:
            return False


class ChromaDB(VectorDatabase):
    """
    An implementation for Chroma vector database.
    """
    CHROMA_DB_PATH = os.path.join(DATA_DIR, "chroma_db")

    @classmethod
    def get(cls,
            embedding_model: Union[HuggingFaceEmbeddings, OpenAIEmbeddings],
            country: str) -> Chroma:
        """
        Retrieve the vector database.

        Parameters
        ----------
        embedding_model: Union[HuggingFaceEmbeddings, OpenAIEmbeddings]
            The embedding model.
        country: str
            The desired country.

        Returns
        -------
        Chroma
            The Chroma database.
        """
        if not cls._check_path_exist(cls.CHROMA_DB_PATH):
            print(f"ChromaDB doesn't exist. Creating ChromaDB.")
            cls._create_db(embedding_model, country)
        print(f"Loading ChromaDB from {cls.CHROMA_DB_PATH}")
        return cls._load_db(embedding_model)

    @classmethod
    def _create_db(cls,
                   embedding_model: Union[HuggingFaceEmbeddings,
                                          OpenAIEmbeddings],
                   country: str):
        """
        Create and save the vector database.

        Parameters
        ----------
        embedding_model: Union[HuggingFaceEmbeddings, OpenAIEmbeddings]
            The embedding model.
        country: str
            The desired country.
        """
        os.mkdir(cls.CHROMA_DB_PATH)
        loader = CSVLoader(
            file_path=os.path.join(DATA_DIR,
                                   f"processed/{country}_processed_df.csv"))
        documents = loader.load()
        Chroma.from_documents(
            documents, embedding_model,
            persist_directory=cls.CHROMA_DB_PATH
        )

    @classmethod
    def _load_db(cls,
                 embedding_model: Union[HuggingFaceEmbeddings,
                                        OpenAIEmbeddings]) -> Chroma:
        """
        To load the vector database.

        Parameters
        ----------
        embedding_model: Union[HuggingFaceEmbeddings, OpenAIEmbeddings]
            The embedding model.

        Returns
        -------
        Chroma
            The Chroma vector database.
        """
        return Chroma(persist_directory=cls.CHROMA_DB_PATH,
                      embedding_function=embedding_model)


class FaissDB(VectorDatabase):
    """
    An implementation for FAISS database.
    """
    FAISS_DB_PATH = os.path.join(DATA_DIR, "faiss_db")

    @classmethod
    def get(cls, embedding_model, country):
        pass

    @classmethod
    def _create_db(cls,
                   embedding_model,
                   country):
        pass

    @classmethod
    def _load_db(cls,
                 embedding_model: Union[HuggingFaceEmbeddings,
                                        OpenAIEmbeddings]):
        pass
