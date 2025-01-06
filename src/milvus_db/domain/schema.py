from typing import List, Dict

import PIL.Image
from pydantic import BaseModel


class InsertRequest(BaseModel):
    collection_name : str = 'test'
    origin_file_names : List[str] | None = None
    images: List[PIL.Image.Image] | None = None
    description : List[str] = ''
    content_type : str = 'not defined'

    meta_info: List[Dict] | None = None

    class Config:
        arbitrary_types_allowed = True


class SearchTextsRequest(BaseModel):
    query: str
    collection_name : str = 'test'

class SearchRequest(BaseModel):
    queries : List[str]
    collection_name: str = 'test'

