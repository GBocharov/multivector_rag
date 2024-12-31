from typing import List, Dict

import PIL.Image
from pydantic import BaseModel


class InsertImages(BaseModel):
    images: List[PIL.Image.Image]
    collection_name : str = 'test'
    origin_file_name : str | None = None
    meta_info: List[Dict] | None = None

    class Config:
        arbitrary_types_allowed = True

class InsertImagesToDB(InsertImages):
    names : List[str]

class InsertTextsRequest(BaseModel):
    texts: List[str]
    collection_name : str = 'test'


class SearchTextsRequest(BaseModel):
    query: str
    collection_name : str = 'test'



class SearchRequest(BaseModel):
    qyerys : List[str]
    collection_name: str = 'test'

