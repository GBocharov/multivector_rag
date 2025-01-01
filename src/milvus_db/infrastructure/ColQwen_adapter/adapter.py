import io
import os

import httpx
from PIL.Image import Image

BASE_URL = os.getenv("SERVICE_URL", "http://col-qwen-llm:8001")

async def send_request(method: str, url: str, **kwargs):
    async with httpx.AsyncClient() as client:
        try:
            if method == 'GET':
                response = await client.get(url, **kwargs)
            elif method == 'POST':
                response = await client.post(url, **kwargs)
            response.raise_for_status()  # Проверка на успешный ответ
            return response.json()
        except httpx.HTTPStatusError as e:
            return {"error": f"HTTP Error: {e.response.status_code}"}
        except httpx.RequestError as e:
            return {"error": f"Request Error: {str(e)}"}

async def get_device():
    url = f"{BASE_URL}/llm_router/get_device_llm_router_llm_info_post"
    return await send_request('GET', url)

async def image_embeddings(image: Image, filename: str = 'image'):
    url = f"{BASE_URL}/llm_router/get_image_embeddings"
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')  # Можно указать другой формат, если нужно
    buffer.seek(0)  # Возврат в начало буфера

    files = {
        "file": (filename, buffer)
    }
    return await send_request('POST', url, files=files)

async def text_embeddings(text: str = 'test'):
    url = f"{BASE_URL}/llm_router/get_text_embeddings?text={text}"
    return await send_request('POST', url)

#
#
# async def image_chat(image: UploadFile, query:str = 'test'):
#     request_object_content = await image.read()
#     img = Image.open(io.BytesIO(request_object_content))
#
#     res = image_query(img, query)
#     return res