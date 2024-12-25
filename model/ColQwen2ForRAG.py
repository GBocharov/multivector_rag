import os
from typing import Any, cast, List

import PIL
import torch
from colpali_engine import ColQwen2, ColQwen2Processor
from colpali_engine.utils.torch_utils import get_torch_device
from peft import LoraConfig
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from PIL.Image import Image

from document_utils.doc_parsers import scale_image


class ColQwen2ForRAG(ColQwen2):
    """
    ColQwen2 model implementation that can be used both for retrieval and generation.
    Allows switching between retrieval and generation modes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_retrieval_enabled = True

    def forward(self, *args, **kwargs) -> Any:
        """
        Forward pass that calls either Qwen2VLForConditionalGeneration.forward for generation
        or ColQwen2.forward for retrieval based on the current mode.
        """
        if self.is_retrieval_enabled:
            return ColQwen2.forward(self, *args, **kwargs)
        else:
            return Qwen2VLForConditionalGeneration.forward(self, *args, **kwargs)

    def generate(self, *args, **kwargs):
        """
        Generate text using Qwen2VLForConditionalGeneration.generate.
        """
        if not self.is_generation_enabled:
            raise ValueError(
                "Set the model to generation mode by calling `enable_generation()` before calling `generate()`."
            )
        return super().generate(*args, **kwargs)

    @property
    def is_retrieval_enabled(self) -> bool:
        return self._is_retrieval_enabled

    @property
    def is_generation_enabled(self) -> bool:
        return not self.is_retrieval_enabled

    def enable_retrieval(self) -> None:
        """
        Switch to retrieval mode.
        """
        self.enable_adapters()
        self._is_retrieval_enabled = True

    def enable_generation(self) -> None:
        """
        Switch to generation mode.
        """
        self.disable_adapters()
        self._is_retrieval_enabled = False


model_name = "vidore/colqwen2-v1.0"
device = get_torch_device("auto")

print(f"Using device: {device}")

# Get the LoRA config from the pretrained retrieval model
lora_config = LoraConfig.from_pretrained(model_name)

# Load the processors
processor_retrieval = cast(ColQwen2Processor, ColQwen2Processor.from_pretrained(model_name))
processor_generation = cast(Qwen2VLProcessor, Qwen2VLProcessor.from_pretrained(lora_config.base_model_name_or_path))

# Load the model with the loaded pre-trained adapter for retrieval
model = cast(
    ColQwen2ForRAG,
    ColQwen2ForRAG.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    ),
)

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
    return result_vectors

def get_text_embeddings(queries : List[str]) -> torch.Tensor:
    result_vectors = []
    batch_size = 1
    batches = [queries[i: i + batch_size] for i in range(len(queries))]

    for batch in batches:
        batch_images = processor_retrieval.process_queries(batch).to(model.device)
        with torch.no_grad():
            image_embeddings = model.forward(**batch_images)
            image_embeddings = list(torch.unbind(image_embeddings.to('cpu')))
            result_vectors += image_embeddings
    return result_vectors


if __name__ == '__main__':
    # image = PIL.Image.open(r'/home/gleb/PycharmProjects/vllm_db_test/pages/page_6.png')
    # text = 'text text text2'
    # image_emb = get_get_image_embeddings([image])
    # text_emb = get_get_text_embeddings([text])
    #
    # print(f'IMAGE \n {image_emb.shape} \n {image_emb}')
    #
    # print(f'TEXT \n {text_emb.shape} \n {text_emb}')

    images = [scale_image(PIL.Image.open('../pages/' + name)) for name in os.listdir('../pages')]
    paths = [name for name in os.listdir("../pages")]
    embeddings = get_image_embeddings(images)
    #embeddings = list(torch.unbind(embeddings.to("cpu")))
    print(embeddings)
    for e in embeddings:
       print(e.shape)

    texts = [
        '1',
        '12',
        '1 2',
        '1 2 3 4',
    ]
    embeddings = get_text_embeddings(texts)
    # embeddings = list(torch.unbind(embeddings.to("cpu")))
    print(embeddings)
    for e in embeddings:
        print(e.shape)