
from typing import Any, cast, List

import torch
from colpali_engine import ColQwen2, ColQwen2Processor
from colpali_engine.utils.torch_utils import get_torch_device
from peft import LoraConfig
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
class ColQwen2ForRAG(ColQwen2):
    """
    ColQwen2 ColQwenLLM implementation that can be used both for retrieval and generation.
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
                "Set the ColQwenLLM to generation mode by calling `enable_generation()` before calling `generate()`."
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


model_name =  "vidore/colqwen2-v1.0"
cache_hub = r"/opt/app-root/cache_hub/clpl"
#mmm = r'/opt/app-root/models/colqwen2-v1.0/checkpoint-2310/'
device = get_torch_device("auto")

print(f"Using device: {device}")


# Get the LoRA config from the pretrained retrieval ColQwenLLM
lora_config = LoraConfig.from_pretrained(model_name,  cache_dir=cache_hub)

print(model_name)
# Load the processors
processor_retrieval = cast(ColQwen2Processor, ColQwen2Processor.from_pretrained(model_name, cache_dir=cache_hub))
processor_generation = cast(Qwen2VLProcessor, Qwen2VLProcessor.from_pretrained(lora_config.base_model_name_or_path, cache_dir=cache_hub))

# Load the ColQwenLLM with the loaded pre-trained adapter for retrieval
model = cast(
    ColQwen2ForRAG,
    ColQwen2ForRAG.from_pretrained(
        pretrained_model_name_or_path = model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        cache_dir=cache_hub
    ),
)



