import os
import pickle
from abc import ABC, abstractmethod
import shutil

from milvus_db.domain.CollectionsBuilder import ColQwenCollection
from milvus_db.infrastructure.ColQwen_adapter.adapter import image_embeddings, text_embeddings
import milvus_db.infrastructure.config as milvus_config
from milvus_db.domain.schema import InsertImages, InsertImagesToDB, SearchRequest


test_retriever = ColQwenCollection(collection_name="test")

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
    async def search(self, entity):
        pass

class MilvusRepository(Repository):

    async def search(self, request: SearchRequest ):
        querys, collection_name = request.qyerys, request.collection_name
        results = []
        print(f'query = {querys}')

        for query in querys:
            response = await text_embeddings(query)
            query = pickle.loads(response.content)[0]
            result = test_retriever.search(query, topk=5)
            results.append(result)
        return results

    async def get_info(self, request = None):
        return test_retriever.collection_name

    #batch insert
    async def insert(self, request: InsertImagesToDB):

        images, names, collection_name = request.images, request.names, request.collection_name


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
            res = test_retriever.insert(data)
            result.append(res)
        return result


    async def delete(self, collection_name:str):
        test_retriever.clear()
        return f'collection {collection_name} successfully deleted'

    async def get(self, request):
        pass



def get_available_save_path(upload_dir_base:str, collection_name:str, origin) -> str:
    counter = 1
    extension = '.png'
    upload_dir = os.path.join(upload_dir_base, collection_name)

    os.makedirs(upload_dir, exist_ok=True)

    filename = f'{upload_dir}_{origin}_{counter}_{extension}'

    while os.path.exists(filename):
        filename = f"{upload_dir}_{origin}_{counter}{extension}"
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
            save_path = get_available_save_path(upload_dir, collection_name, request.origin_file_name)
            save_paths.append(save_path)
            image.save(save_path)

        return save_paths

    async def delete(self, collection_name: str):
        upload_dir = os.path.join(milvus_config.milvus_image_data_save_dir, collection_name)
        if os.path.exists(upload_dir):
            shutil.rmtree(upload_dir)
        return f'collection {collection_name} files successfully deleted'

    async def get(self, entity):
        pass

    async def info(self, entity):
        pass


class UnitOfWork:
    pass