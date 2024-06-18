from enum import Enum


class ToolType(str, Enum):
    RETRIEVER = "retriever-tool"
    ONLINE_SEARCH = "online-search-tool"
