import torch

from colpali_engine.utils.torch_utils import get_torch_device
import sys

print("Версия Python:", sys.version)

# Вывод версии PyTorch
torch_version = torch.__version__
print("PyTorch version:", torch_version)
# Вывод версии CUDA
cuda_version = torch.version.cuda
print("CUDA version:", cuda_version)
device = get_torch_device("auto")
print(f"Using device: {device}")

# sudo docker run --rm -ti --runtime=nvidia     -e NVIDIA_VISIBLE_DEVICES=all    my-python-app !!!!!!!!!!!!!!!
