
from typing import List

import torch
from PIL.Image import Image

from ColQwenLLM.ColQwen2ForRAG import processor_retrieval, model, processor_generation, device
from document_utils.doc_parsers import scale_image


def get_image_embeddings(images : List[Image]) -> List[torch.Tensor]:

    result_vectors = []
    batch_size = 1
    batches = [images[i: i + batch_size] for i in range(len(images))]

    for batch in batches:
        batch_images = processor_retrieval.process_images(batch).to(model.device)
        with torch.no_grad():
            image_embeddings = model.forward(**batch_images)
            image_embeddings = list(torch.unbind(image_embeddings.to('cpu')))
            result_vectors += image_embeddings
    return [v.float().numpy() for v in result_vectors]

def get_text_embeddings(queries : List[str]) :
    result_vectors = []
    batch_size = 1
    batches = [queries[i: i + batch_size] for i in range(len(queries))]

    for batch in batches:
        batch_images = processor_retrieval.process_queries(batch).to(model.device)
        with torch.no_grad():
            image_embeddings = model.forward(**batch_images)
            image_embeddings = list(torch.unbind(image_embeddings.to('cpu')))
            result_vectors += image_embeddings
    return [v.float().numpy() for v in result_vectors]

def image_query(image: Image, query: str = 'test'):
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": f"Перед тобой смешная картинка с текстом: {query}",
                },
            ],
        }
    ]

    image = scale_image(image, 624)
    text_prompt = processor_generation.apply_chat_template(conversation, add_generation_prompt=True)
    inputs_generation = processor_generation(
        text=[text_prompt],
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to(device)

    # Generate the RAG response
    model.enable_generation()
    output_ids = model.generate(**inputs_generation, max_new_tokens=128)

    # Ensure that only the newly generated token IDs are retained from output_ids
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                     zip(inputs_generation.input_ids, output_ids)]

    # Decode the RAG response
    output_text = processor_generation.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    model.enable_retrieval()
    return output_text

