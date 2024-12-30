import os
import pickle
from abc import ABC, abstractmethod

from pymilvus import MilvusClient

from milvus_db.domain.MilvusColbertCollection import MilvusColbertCollection
from milvus_db.external.llm_response import image_embeddings, text_embeddings
import milvus_db.infrastructure.config as milvus_config
from milvus_db.infrastructure.schema import InsertImages, InsertImagesToDB, SearchRequest

client = MilvusClient(milvus_config.milvus_db_save_dir + '/' +"milvus_demo.db")
#client.insert()
test_retriever = MilvusColbertCollection(collection_name="test", milvus_client=client)

collections = {
    'test' : test_retriever
}

class Repository(ABC):

    @abstractmethod
    async def get_info(self, entity):
        pass

    @abstractmethod
    async def insert(self, entity):
        pass

    @abstractmethod
    async def delete(self, entity):
        pass

    @abstractmethod
    async def get(self, entity):
        pass

    @abstractmethod
    async def info(self, entity):
        pass

    @abstractmethod
    async def search(self, entity):
        pass

class MilvusRepository(Repository):

    async def search(self, request: SearchRequest ):

        querys, collection_name = request.qyerys, request.collection_name
        retriever = collections[collection_name]
        results = []
        print(f'query = {querys}')

        for query in querys:
            response = await text_embeddings(query)
            query = pickle.loads(response.content)[0]
            result = retriever.search(query, topk=5)
            results.append(result)
            # import pprint as pp
            # print('------>')
            # # pp.pprint(results)
            # pp.pprint(type(results))
        return results

    async def get_info(self, request = None):
        list_collections = client.list_collections()
        return list_collections

    #batch insert
    async def insert(self, request: InsertImagesToDB):

        images, names, collection_name = request.images, request.names, request.collection_name

        retriever = collections[collection_name]

        result = []

        for i in range(len(images)):
            response = await image_embeddings(images[i])

            print(f'R E S P O N S E = = = ={response}')
            embedding = pickle.loads(response.content)  # embedding = await image_embeddings(images[i])
            print(embedding)
            data = {
                "colbert_vecs": embedding[0],
                "doc_id": i,
                "filepath": names[i] if names else '',
            }
            res = retriever.insert(data)
            result.append(res)
        return result


    async def delete(self, collection_name:str):
        if collection_name not in client.list_collections():
            return 'no such collection'
        client.drop_collection(collection_name=collection_name)
        collections[collection_name] = MilvusColbertCollection(collection_name="test", milvus_client=client)
        return f'collection {collection_name} successfully deleted'

    async def get(self, request):
        pass

    async def info(self, request):
        pass


def get_available_save_path(upload_dir_base:str, collection_name:str) -> str:
    counter = 1
    extension = '.png'
    upload_dir = os.path.join(upload_dir_base, collection_name)

    os.makedirs(upload_dir, exist_ok=True)

    filename = f'{upload_dir}_{counter}_{extension}'

    while os.path.exists(filename):
        filename = f"{upload_dir}_{counter}{extension}"
        counter += 1
    return filename

class FileSystemRepository(Repository):

    async def search(self, entity):
        pass

    async def get_info(self, entity):
        pass

    async def insert(self, request: InsertImages):
        images, collection_name, origin_file = request.images, request.collection_name, request.origin_file_name

        upload_dir = os.path.join(milvus_config.milvus_image_data_save_dir, collection_name)
        save_paths = []
        for image in images:
            save_path = get_available_save_path(upload_dir, collection_name)
            save_paths.append(save_path)
            image.save(save_path)

        return save_paths

    async def delete(self, collection_name: str):
        if collection_name not in client.list_collections():
            return 'no such collection'
        upload_dir = os.path.join(milvus_config.milvus_image_data_save_dir, collection_name)
        os.remove(upload_dir)
        return f'collection {collection_name} files successfully deleted'

    async def get(self, entity):
        pass

    async def info(self, entity):
        pass


class UnitOfWork:
    pass