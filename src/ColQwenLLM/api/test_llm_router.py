import io
from fastapi import FastAPI
import pytest
import pickle
from fastapi.testclient import TestClient
from PIL import Image
from io import BytesIO

import httpx

BASE_URL = "http://localhost:8001"


# Helper function to create a dummy image
def create_dummy_image():
    img = Image.new('RGB', (100, 100), color = 'red')
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr

# Test for "/llm_info" endpoint
@pytest.mark.asyncio
async def test_get_device():
    async with httpx.AsyncClient() as session:
        response = await session.get(f"{BASE_URL}/llm_router/llm_info")
    
    assert response.status_code == 200
    assert 'device' in response.json()

# Test for "/get_image_embeddings" endpoint
@pytest.mark.asyncio
async def test_image_embeddings():
    image_file = create_dummy_image()
    files = {
        "file": ('test_im', image_file)
    }
    async with httpx.AsyncClient() as session:
        response = await session.post(
            f"{BASE_URL}/llm_router/get_image_embeddings",
            files=files
            )
    assert response.status_code == 200
    # Check if the response is a valid pickle object
    assert pickle.loads(response.content)  # If this raises an error, the response is not a valid pickle object

# Test for "/get_text_embeddings" endpoint
@pytest.mark.asyncio
async def test_text_embeddings():
    text = 'test'
    message = {"message": text}
    async with httpx.AsyncClient() as session:
        response = await session.post(
            f"{BASE_URL}/llm_router/get_text_embeddings",
            json=message
            )
    assert response.status_code == 200
    # Check if the response is a valid pickle object
    assert pickle.loads(response.content)  # If this raises an error, the response is not a valid pickle object

