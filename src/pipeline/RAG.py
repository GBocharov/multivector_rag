from abc import ABC, abstractmethod


class VecRAG(ABC):

    @abstractmethod
    def text_search(self, query: str = None):
        pass

    @abstractmethod
    def text_search_generation(self, query: str = None):
        pass


class MilvusRAG(VecRAG):
    def __init__(self, retriever, model):
        pass

    def text_search(self, query: str = None):
        pass

    def text_search_generation(self, query: str = None):
        pass