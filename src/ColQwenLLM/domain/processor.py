import logging.config
from typing import List

import numpy as np
import torch
from PIL.Image import Image

from ColQwenLLM.domain.ColQwen2ForRAG import processor_retrieval, model, processor_generation, device
from ColQwenLLM.paths_config import logger_conf_path
from ColQwenLLM.prompts.image_conversation import format_prompt
from document_utils.doc_parsers import scale_image


logging.config.fileConfig(logger_conf_path)
logger = logging.getLogger('llmLogger')


async def get_image_embeddings(images: List[Image], batch_size: int = 1) -> List[np.ndarray]:
    result_vectors = []

    batches = [images[i:i + batch_size] for i in range(0, len(images), batch_size)]

    for batch_index, batch in enumerate(batches):
        try:
            batch_images = processor_retrieval.process_images(batch).to(model.device)

            with torch.no_grad():
                image_embeddings = model.forward(**batch_images)
                result_vectors.extend(embedding.cpu().float().numpy() for embedding in torch.unbind(image_embeddings))

        except Exception as e:
            logger.error(f"Error processing batch #{batch_index + 1} (size {len(batch)}): {e}")
            continue

    return result_vectors


async def get_text_embeddings(queries: List[str], batch_size: int = 1) -> List[np.ndarray]:
    result_vectors = []

    batches = [queries[i:i + batch_size] for i in range(0, len(queries), batch_size)]

    for batch_index, batch in enumerate(batches):
        try:
            batch_texts = processor_retrieval.process_queries(batch).to(model.device)

            with torch.no_grad():
                text_embeddings = model.forward(**batch_texts)

                text_embeddings = list(torch.unbind(text_embeddings.to('cpu')))
                result_vectors += text_embeddings

        except Exception as e:
            logger.error(f"Error processing batch #{batch_index + 1} (size {len(batch)}): {e}")
            # Optionally continue to the next batch or handle the error as needed
            continue

    # Convert results to float numpy arrays
    return [v.float().numpy() for v in result_vectors]


async def _prepare_inputs(text_prompt: str, image: Image) -> dict:
    """Prepare model inputs from the text prompt and image."""
    try:

        scaled_image = scale_image(image, 624)
        return processor_generation(
            text=[text_prompt],
            images=[scaled_image],
            padding=True,
            return_tensors="pt",
        ).to(device)
    except Exception as e:
        logger.error("Error preparing inputs for the model: %s", e, exc_info=True)
        raise

async def _generate_model_response(inputs_generation: dict) -> List[str]:
    """Generate a response from the model using the provided inputs."""
    try:

        model.enable_generation()
        output_ids = model.generate(**inputs_generation, max_new_tokens=128)

        # Ensure that only the newly generated token IDs are retained from output_ids
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                         zip(inputs_generation.input_ids, output_ids)]

        output_text = processor_generation.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        model.enable_retrieval()
        return output_text
    except Exception as e:
        logger.error("Error generating response from the model: %s", e, exc_info=True)
        raise


async def image_query(image: Image, query: str = 'test'):
    try:
        prompt = format_prompt(query)
        image = scale_image(image, 624)
        text_prompt = processor_generation.apply_chat_template(prompt, add_generation_prompt=True)
        inputs_generation = await _prepare_inputs(text_prompt, image)
        output_text = await _generate_model_response(inputs_generation)
        return output_text

    except Exception as e:
        logger.error("An error occurred during the image query process.", exc_info=True)
        return str(e)  # Optionally return the error message

