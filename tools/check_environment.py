"""Small GPU environment check; run inside a SLURM GPU allocation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

import numpy as np
import torch
import torchvision
import dataset
import models
from nms.cpu_nms import cpu_nms
from nms.gpu_nms import gpu_nms
from torch.utils.tensorboard import SummaryWriter

print("Python:", sys.version, flush=True)
print("PyTorch:", torch.__version__, "torchvision:", torchvision.__version__, flush=True)
print("CUDA runtime:", torch.version.cuda, flush=True)
assert torch.cuda.is_available(), "No CUDA GPU available; run in a GPU allocation"
print("GPU:", torch.cuda.get_device_name(0), flush=True)
x = torch.ones((32, 32), device="cuda")
assert (x @ x).sum().item() == 32768
boxes = np.array([[0, 0, 10, 10, 0.9], [0, 0, 10, 10, 0.8]], dtype=np.float32)
assert list(cpu_nms(boxes, 0.5)) == [0]
assert list(gpu_nms(boxes, 0.5, 0)) == [0]
print("PASS: imports, CUDA matmul, CPU/GPU NMS", flush=True)
