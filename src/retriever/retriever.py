from abc import ABC


class RAGRetriever(ABC):

    def create_collection(self):
       pass

    def search(self, data, topk):
        pass

    def insert(self, data):
        pass

    def delete_rows(self, data):
        pass