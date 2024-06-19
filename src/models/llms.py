from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFaceEndpoint
from .model_type import ModelType
from typing import Union

REGISTRY_MODEL = {
    ModelType.CHATGPTSTANDARD: "gpt-3.5-turbo",
    ModelType.PHITHREE4k: "microsoft/Phi-3-mini-4k-instruct"
}


class Models:
    """
    A class to get the LLM.
    """
    @classmethod
    def get(cls, model_name: ModelType) -> Union[ChatOpenAI,
                                                 HuggingFaceEndpoint]:
        """
        Retrieve the desired LLm.

        Parameters
        ----------
        model_name: ModelType
            The name of the LLM.

        Returns
        -------
        Union[ChatOpenAI, HuggingFaceEndpoint]
            The desired LLM.

        Raises
        ------
        NotImplementedError
            If the desired LLM has not been implemented.
        """
        if model_name == ModelType.CHATGPTSTANDARD:
            print(f"[INFO] Using {ModelType.CHATGPTSTANDARD}")
            return ChatOpenAI(
                model_name=REGISTRY_MODEL[model_name],
                temperature=0
            )
        elif model_name == ModelType.PHITHREE4k:
            print(f"[INFO] Using {ModelType.PHITHREE4k}")
            endpoint = f"https://api-inference.huggingface.co/models/" \
                       f"{REGISTRY_MODEL[model_name]}"
            return HuggingFaceEndpoint(
                endpoint_url=endpoint,
                task="text-generation",
                temmperature=0.1
            )
        else:
            raise NotImplementedError("Other LLM have not been implemented.")
