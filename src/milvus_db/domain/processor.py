from pymilvus import MilvusClient

from milvus_db.domain.Repository import MilvusRepository, FileSystemRepository
from milvus_db.domain.schema import InsertImages, InsertImagesToDB, SearchRequest



async def get_db_info(session, db_client: MilvusClient):
    collections_names = await MilvusRepository.get_info(db_client)
    return collections_names

async def get_collection_info(session, db_client: MilvusClient, collection_name:str):
    collections_names = await MilvusRepository.get_info(db_client)
    return collections_names


def drop_db(session, db_client: MilvusClient):
    #milvus_repository.delete('')
        return None

async def drop_collection(session, db_client: MilvusClient, collection_name:str):

    r1 = await MilvusRepository.delete(db_client=db_client, collection_name=collection_name)
    r2 = await FileSystemRepository.delete(collection_name=collection_name)

    return r1, r2

async def insert_Images(session, db_client: MilvusClient, request: InsertImages):

    save_paths = await FileSystemRepository.insert(request)

    insert_request = InsertImagesToDB(
        images = request.images,
        collection_name = request.collection_name,
        names = save_paths,
        origin_file_name = None,
        meta_info = None
    )
    await MilvusRepository.insert(session, db_client, insert_request)

    return save_paths


async def search_Texts(
        session,
        db_client: MilvusClient,
        request:SearchRequest
):
    results = await MilvusRepository.search(session, db_client, request)
    return results