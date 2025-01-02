import os
import pickle
import shutil
from abc import ABC, abstractmethod


import milvus_db.infrastructure.config as milvus_config
from milvus_db.domain.CollectionsBuilder import ColQwenCollection
from milvus_db.infrastructure.ColQwen_adapter.adapter import image_embeddings, text_embeddings
from milvus_db.domain.schema import InsertImages, InsertImagesToDB, SearchRequest


class Repository(ABC):

    @classmethod
    @abstractmethod
    async def get_info(cls, *args, **kwargs):
        pass

    @classmethod
    @abstractmethod
    async def insert(cls, *args, **kwargs):
        pass

    @classmethod
    @abstractmethod
    async def delete(cls, *args, **kwargs):
        pass

    @classmethod
    @abstractmethod
    async def get(cls, *args, **kwargs):
        pass

    @classmethod
    @abstractmethod
    async def search(cls, *args, **kwargs):
        pass


class MilvusRepository(Repository):

    @classmethod
    async def search(cls, session, db_client, request: SearchRequest):
        queries, collection_name = request.qyerys, request.collection_name
        results = []
        print(f'query = {queries}')
        with db_client as cl:
            for query in queries:
                response = await text_embeddings(session, query)

                print(f'R E S P O N S E = = = ={response}')
                query = pickle.loads(response.content)[0]
                result = ColQwenCollection.search(cl, collection_name, query, topk=5)
                results.append(result)
        return results


    #TODO batch insert
    @classmethod
    async def insert(cls, session, db_client, request: InsertImagesToDB):

        images, names, collection_name = request.images, request.names, request.collection_name

        result = []
        with db_client as cl:
            for i in range(len(images)):
                response = await image_embeddings(session, images[i])

                print(f'R E S P O N S E = = = ={response}')
                embedding = pickle.loads(response.content)  # embedding = await image_embeddings(images[i])
                print(embedding)
                data = {
                    "colbert_vecs": embedding[0],
                    "doc_id": i,
                    "filepath": names[i] if names else '',
                }
                res = ColQwenCollection.insert(cl, collection_name, data)
                result.append(res)
        return result

    @classmethod
    async def delete(cls, db_client, collection_name:str):

        with db_client as cl:
            ColQwenCollection.clear(cl, collection_name)
        return f'collection {collection_name} successfully deleted'

    @classmethod
    async def get(cls, db_client, collection_name:str, request):
        pass

    @classmethod
    async def get_info(cls, db_client):
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

    @classmethod
    async def search(cls, entity):
        pass

    @classmethod
    async def get_info(cls, entity):
        pass

    @classmethod
    async def insert(cls, request: InsertImages):
        images, collection_name, origin_file = request.images, request.collection_name, request.origin_file_name
        upload_dir = os.path.join(milvus_config.milvus_image_data_save_dir, collection_name)
        save_paths = []
        for image in images:
            save_path = get_available_save_path(upload_dir, collection_name, request.origin_file_name)
            save_paths.append(save_path)
            image.save(save_path)
        return save_paths

    @classmethod
    async def delete(cls, collection_name: str):
        upload_dir = os.path.join(milvus_config.milvus_image_data_save_dir, collection_name)
        if os.path.exists(upload_dir):
            shutil.rmtree(upload_dir)
        return f'collection {collection_name} files successfully deleted'

    @classmethod
    async def get(cls, entity):
        pass

    @classmethod
    async def info(cls, entity):
        pass


class UnitOfWork:
    pass