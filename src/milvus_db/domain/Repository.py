import asyncio
import os
import pickle
import shutil
import logging.config
from abc import ABC, abstractmethod
from typing import List

from milvus_db.domain.schema import SearchRequest, InsertRequest
import milvus_db.infrastructure.config as milvus_config
from milvus_db.domain.CollectionsBuilder import ColQwenCollection
from milvus_db.infrastructure.ColQwen_adapter.adapter import image_embeddings, text_embeddings

# Настройка логирования
logging.config.fileConfig(milvus_config.logger_conf_path)
logger = logging.getLogger('milvusLogger')



class Repository(ABC):

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
    async def search(cls, *args, **kwargs):
        pass


class MilvusRepository(Repository):

    @classmethod
    async def search(cls, session, db_client, collection_name, queries):
        results = []
        logger.info(f"Starting search in collection: {collection_name} with queries: {queries}")
        try:
            for query in queries:
                response = await text_embeddings(session, query)
                logger.debug(f"Embedding response: {response}")
                query = pickle.loads(response.content)[0]
                result = ColQwenCollection.search(db_client, collection_name, query, topk=5)
                results.append(result)
        except Exception as e:
            logger.error(f"Error during search: {e}", exc_info=True)
            raise
        logger.info(f"Search completed with results: {results}")
        return results


    @classmethod
    async def delete(cls, db_client, collection_name: str, origin_file_names: List[str] = None):
        logger.info(f"Starting delete in collection: {collection_name}")
        try:
            if origin_file_names:
                filter = f'origin_file_names in [{origin_file_names}]'
                ColQwenCollection.delete(db_client, collection_name, filter)
            else:
                ColQwenCollection.delete(db_client, collection_name)
        except Exception as e:
            logger.error(f"Error during delete: {e}", exc_info=True)
            raise

        logger.info(f"Collection {collection_name} successfully deleted")
        return f'collection {collection_name} successfully deleted'

    @classmethod
    async def insert(cls, session, db_client, collection_name, images, names):
        logger.info(f"Starting insert into collection: {collection_name}")
        inserted_doc_pks = []
        try:
            # Получаем эмбеддинги
            embeddings = await cls._fetch_embeddings(session, images)

            # Создаём и вставляем данные
            results, inserted_doc_pks = cls._insert_data(db_client, collection_name, embeddings, names)
        except Exception as e:
            logger.error(f"Error during insert: {e}", exc_info=True)
            if inserted_doc_pks:
                cls._rollback_insert(db_client, collection_name, inserted_doc_pks)
            raise

        logger.info(f"Insert completed with result: {results}")
        return results

    @classmethod
    async def _fetch_embeddings(cls, session, images):
        """Получение эмбеддингов для изображений параллельно."""
        try:
            responses = await asyncio.gather(*(image_embeddings(session, image) for image in images))
            embeddings = [pickle.loads(response.content) for response in responses]
            logger.debug(f"Fetched embeddings for {len(images)} images")
            return embeddings
        except Exception as e:
            logger.error(f"Error fetching embeddings: {e}", exc_info=True)
            raise

    @classmethod
    def _insert_data(cls, db_client, collection_name, embeddings, names):
        """Создание данных для вставки и добавление в коллекцию."""
        results = []
        inserted_doc_pks = []

        for i, embedding in enumerate(embeddings):
            data = {
                "colbert_vecs": embedding[0],
                "doc_id": i,
                "source_path": names[i] if names else '',
                "content_type": "image",
                "description": '',
                "meta_info": None
            }

            res = ColQwenCollection.insert(db_client, collection_name, data)
            inserted_doc_pks.append(res["ids"])
            results.append(res)

        logger.debug(f"Inserted {len(results)} documents into collection {collection_name}")
        return results, inserted_doc_pks

    @classmethod
    def _rollback_insert(cls, db_client, collection_name, inserted_doc_pks):
        """Откат вставленных данных в случае ошибки."""
        logger.warning(f"Rolling back inserted documents: {inserted_doc_pks}")
        filter = f'pk in [{", ".join(map(str, inserted_doc_pks))}]'
        ColQwenCollection.delete(db_client, collection_name, filter)
        logger.info("Rollback completed.")



class FileSystemRepository(Repository):

    @classmethod
    async def search(cls, entity):
        pass

    @classmethod
    async def insert(cls, collection_name, images, origin_file_name):
        upload_dir = os.path.join(milvus_config.milvus_image_data_save_dir, collection_name)
        save_paths = []
        logger.info(f"Starting file save for collection: {collection_name}")

        try:
            for image in images:
                save_path = cls._get_available_save_path(upload_dir, collection_name, origin_file_name)
                save_paths.append(save_path)
                image.save(save_path)  # Здесь может возникнуть ошибка
                logger.debug(f"Saved image to {save_path}")
        except Exception as e:
            logger.error(f"Error during saving images: {e}", exc_info=True)
            for path in save_paths:
                if os.path.exists(path):
                    os.remove(path)
                    logger.warning(f"Removed file due to error: {path}")
            raise

        logger.info(f"File save completed with paths: {save_paths}")
        return save_paths

    @classmethod
    async def delete(cls, collection_name: str, origin_file_names: List[str] = None):
        upload_dir = os.path.join(milvus_config.milvus_image_data_save_dir, collection_name)
        logger.info(f"Starting file delete for collection: {collection_name}")

        try:
            if origin_file_names:
                for filename in origin_file_names:
                    full_path = os.path.join(upload_dir, filename)
                    if os.path.exists(full_path):
                        os.remove(full_path)
                        logger.debug(f"Deleted file: {full_path}")
            else:
                if os.path.exists(upload_dir):
                    shutil.rmtree(upload_dir)
                    logger.debug(f"Deleted directory: {upload_dir}")
        except Exception as e:
            logger.error(f"Error during file delete: {e}", exc_info=True)
            raise

        logger.info(f"File delete completed for collection: {collection_name}")
        return f'collection {collection_name} files successfully deleted'

    @classmethod
    def _get_available_save_path(cls, upload_dir_base: str, collection_name: str, origin) -> str:
        counter = 1
        extension = '.png'
        upload_dir = os.path.join(upload_dir_base, collection_name)

        os.makedirs(upload_dir, exist_ok=True)

        filename = os.path.join(upload_dir, f'{origin}_{counter}{extension}')

        while os.path.exists(filename):
            counter += 1
            filename = os.path.join(upload_dir, f"{origin}_{counter}{extension}")

        return filename