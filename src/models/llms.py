from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFaceEndpoint
from .model_type import ModelType

REGISTRY_MODEL = {
    ModelType.CHATGPTSTANDARD: "gpt-3.5-turbo",
    ModelType.PHITHREE4k: "microsoft/Phi-3-mini-4k-instruct"
}


class Models:
    @classmethod
    def get(cls, model_name):
        if model_name == ModelType.CHATGPTSTANDARD:
            print(f"Using {ModelType.CHATGPTSTANDARD}")
            return ChatOpenAI(
                model_name=REGISTRY_MODEL[model_name],
                temperature=0
            )
        elif model_name == ModelType.PHITHREE4k:
            endpoint = f"https://api-inference.huggingface.co/models/" \
                       f"{REGISTRY_MODEL[model_name]}"
            return HuggingFaceEndpoint(
                endpoint_url=endpoint,
                task="text-generation",
                temmperature=0.1
            )
