import os
from typing import List
import logging.config
import PIL.Image
from pymilvus import MilvusClient

from milvus_db.domain.Repository import MilvusRepository, FileSystemRepository
from milvus_db.infrastructure.config import logger_conf_path

logging.config.fileConfig(logger_conf_path)
logger = logging.getLogger('milvusLogger')

async def insert_images_to_collection(session, db_client: MilvusClient, collection_name, images: List[PIL.Image.Image],
                                      origin_file_name: str = None):
    save_paths = []
    try:
        save_paths = await FileSystemRepository.insert(collection_name, images, origin_file_name)

        await MilvusRepository.insert(session, db_client, collection_name, images, save_paths)
    except Exception as e:
        logger.error(f"Error during insertion: {e}", exc_info=True)

        if save_paths:
            logger.warning(f"Rolling back file system changes for paths: {save_paths}")
            for path in save_paths:
                if os.path.exists(path):
                    os.remove(path)

        raise
    return save_paths


async def delete_from_collection(session, db_client: MilvusClient, collection_name: str, paths: List[str] = None):
    try:
        await MilvusRepository.delete(db_client=db_client, collection_name=collection_name, origin_file_names=paths)

        await FileSystemRepository.delete(collection_name=collection_name, origin_file_names=paths)
    except Exception as e:
        logger.error(f"Error during deletion: {e}", exc_info=True)
        raise

    return f"Collection {collection_name} successfully cleaned up"


async def search_texts_data_in_collection(session, db_client: MilvusClient, collection_name, request: str):
    try:
        results = await MilvusRepository.search(session, db_client, collection_name, request)
    except Exception as e:
        logger.error(f"Error during search: {e}", exc_info=True)
        raise
    return results
