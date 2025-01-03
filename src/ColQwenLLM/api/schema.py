import logging.config
from typing import List

from pydantic import BaseModel

from ColQwenLLM.paths_config import logger_conf_path

logging.config.fileConfig(logger_conf_path)
logger = logging.getLogger('llmLogger')

class Message(BaseModel):
    message: str | None = 'test'


class ImageEmbeddingResponse(BaseModel):
    embeddings: List[float]  # или другой тип, в зависимости от того, что вы возвращаете

class ChatResponse(BaseModel):
    response: str