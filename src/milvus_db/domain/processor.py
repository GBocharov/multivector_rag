from milvus_db.domain.Repository import MilvusRepository, FileSystemRepository
from milvus_db.domain.schema import InsertImages, InsertImagesToDB, SearchRequest

milvus_repository = MilvusRepository()
file_system_repository = FileSystemRepository()

async def get_db_info():
    collections_names = await milvus_repository.get_info()
    return collections_names

async def get_collection_info(collection_name:str):

    collections_names = await milvus_repository.get_info()
    return collections_names


def drop_db():
    #milvus_repository.delete('')
        return None

async def drop_collection(collection_name:str):

    r1 = await milvus_repository.delete(collection_name=collection_name)
    r2 = await file_system_repository.delete(collection_name=collection_name)

    return r1, r2

async def insert_Images(request: InsertImages):

    save_paths = await file_system_repository.insert(request)

    insert_request = InsertImagesToDB(
        images = request.images,
        collection_name = request.collection_name,
        names = save_paths,
        origin_file_name = None,
        meta_info = None
    )
    await milvus_repository.insert(insert_request)

    return save_paths


async def search_Texts(
        request:SearchRequest
):
    results = await milvus_repository.search(request)
    return results