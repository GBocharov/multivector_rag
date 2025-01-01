import io
import pickle
import logging.config
from functools import wraps
from typing import Annotated, Callable

from PIL import Image
from fastapi import APIRouter, UploadFile, HTTPException, Depends
from starlette.responses import StreamingResponse

from ColQwenLLM.domain.ColQwen2ForRAG import device
from ColQwenLLM.domain.processor import image_query, get_image_embeddings, get_text_embeddings
from ColQwenLLM.paths_config import logger_conf_path

logging.config.fileConfig(logger_conf_path)
logger = logging.getLogger('llmLogger')

llm_router = APIRouter(
    prefix="/llm_router",
    tags=["LLM"],
)

def exception_handler(func: Callable) -> Callable:
    @wraps(func)
    async def wrapper(*args, **kwargs):

        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"An error occurred in {func.__name__}.", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    return wrapper

async def file_to_image(file: UploadFile) -> Image.Image:
    request_object_content = await file.read()
    try:
        img = Image.open(io.BytesIO(request_object_content))
        return img
    except Exception:
        logger.error("Invalid image file", exc_info=True)
        raise HTTPException(status_code=400, detail="Invalid image file")


@llm_router.post("/llm_info", summary="Get device information")
@exception_handler
async def get_device():
    return device

@llm_router.post("/get_image_embeddings", summary="Get image embeddings")
@exception_handler
async def image_embeddings(image: Annotated[Image.Image, Depends(file_to_image)]):
    res = await get_image_embeddings([image])
    byte_data = pickle.dumps(res)
    logger.info("Successfully processed image embeddings.")
    return StreamingResponse(io.BytesIO(byte_data), media_type="application/octet-stream")

@llm_router.post("/get_text_embeddings", summary="Get text embeddings")
@exception_handler
async def text_embeddings(text: str = 'test'):
    res = await get_text_embeddings([text])
    byte_data = pickle.dumps(res)
    logger.info("Successfully processed text embeddings.")
    return StreamingResponse(io.BytesIO(byte_data), media_type="application/octet-stream")

@llm_router.post("/chat", summary="Chat with image")
@exception_handler
async def image_chat(image: Annotated[Image.Image, Depends(file_to_image)], query: str = 'test'):
    res = await image_query(image, query)
    logger.info("Successfully processed image chat.")
    return res