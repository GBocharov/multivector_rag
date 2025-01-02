import logging.config
from pydantic import BaseModel

from ColQwenLLM.paths_config import logger_conf_path

logging.config.fileConfig(logger_conf_path)
logger = logging.getLogger('llmLogger')

class Message(BaseModel):
    message: str | None = 'test'


