from fastapi import UploadFile
from pydantic import BaseModel


class Chat(BaseModel):
    image: UploadFile
    query: str = 'test'