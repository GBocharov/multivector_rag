from contextlib import contextmanager
from typing import Generator

import aiohttp
from pymilvus import MilvusClient

import milvus_db.infrastructure.config as milvus_config


#TODO разнести по папкам

async def get_client_session() -> Generator[aiohttp.ClientSession, None, None]:
    async with aiohttp.ClientSession() as session:
        yield session


@contextmanager
def get_milvus_client(token: str | None = None) -> MilvusClient:

    if token:
        client = MilvusClient(milvus_config.milvus_db_save_dir + '/' + "test.db", token = token)
    else:
        client = MilvusClient(milvus_config.milvus_db_save_dir + '/' + "milvus_demo.db") # root
    yield client

    client.close()