from langchain_community.vectorstores import Chroma
from langchain.document_loaders.csv_loader import CSVLoader
import os
from abc import abstractmethod

DATA_DIR = os.path.join(os.path.dirname(os.getcwd()), 'data')


class VectorDatabase:
    @classmethod
    @abstractmethod
    def get(cls, embedding_model, country):
        pass

    @staticmethod
    def check_path_exist(path):
        return os.path.exists(path) or os.listdir(path)


class ChromaDB(VectorDatabase):
    CHROMA_DB_PATH = os.path.join(DATA_DIR, "chroma_db")

    @classmethod
    def get(cls, embedding_model, country):
        if not cls.check_path_exist(cls.CHROMA_DB_PATH):
            print(f"ChromaDB doesn't exist. Creating ChromaDB.")
            cls.create_db(embedding_model, country)
        print(f"Loading ChromaDB from {cls.CHROMA_DB_PATH}")
        return cls.load_db(embedding_model)

    @classmethod
    def create_db(cls, embedding_model, country):
        os.mkdir(cls.CHROMA_DB_PATH)
        loader = CSVLoader(
            file_path= os.path.join(DATA_DIR,
                                    f"processed/{country}_processed_df.csv"))
        documents = loader.load()
        Chroma.from_documents(
            documents, embedding_model,
            persist_directory=cls.CHROMA_DB_PATH
        )

    @classmethod
    def load_db(cls, embedding_model):
        return Chroma(persist_directory=cls.CHROMA_DB_PATH,
                      embedding_function=embedding_model)


class FaissDB(VectorDatabase):
    FAISS_DB_PATH = ""

    @classmethod
    def get(cls, embedding_model, country):
        pass
