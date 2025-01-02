import io
import os

import httpx
from PIL.Image import Image

BASE_URL = os.getenv("SERVICE_URL", "http://col-qwen-llm:8001")

async def send_request(session, method: str, url: str, **kwargs):
    try:
        if method == 'GET':
            response = await session.get(url, **kwargs)
        elif method == 'POST':
            response = await session.post(url, **kwargs)
        response.raise_for_status()  # Проверка на успешный ответ
        return response
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP Error: {e.response.status_code}"}
    except httpx.RequestError as e:
        return {"error": f"Request Error: {str(e)}"}

async def get_device(session):
    url = f"{BASE_URL}/llm_router/get_device_llm_router_llm_info_post"
    return await send_request(session,'GET', url)

async def image_embeddings(session, image: Image, filename: str = 'image'):
    url = f"{BASE_URL}/llm_router/get_image_embeddings"
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')  # Можно указать другой формат, если нужно
    buffer.seek(0)  # Возврат в начало буфера

    files = {
        "file": (filename, buffer)
    }
    return await send_request(session, 'POST', url, files=files)

async def text_embeddings(session, text: str = 'test'):
    url = f"{BASE_URL}/llm_router/get_text_embeddings"
    message = {"message": text}

    return await send_request(session,'POST', url, json=message)

#
#
# async def image_chat(image: UploadFile, query:str = 'test'):
#     request_object_content = await image.read()
#     img = Image.open(io.BytesIO(request_object_content))
#
#     res = image_query(img, query)
#     return res