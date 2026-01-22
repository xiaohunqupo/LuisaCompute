import sys
if len(sys.argv) < 2:
    print('Must input backend')
    exit(1)
if sys.argv[1] != 'dx' and sys.argv[1] != 'vk':
    print('Backend only support dx and vk')
    exit(1)
from luisa import *
from luisa.builtin import *
from luisa.types import *
from luisa.util import *
import numpy as np
import logging
import torch
init(sys.argv[1])
if not torch.cuda.is_available():
    logging.error("CUDA environment unavailable.")
    exit(1)
device = torch.device("cuda")

arr = np.zeros(16, dtype=np.float32)
interop_buffer = Buffer(16, float, enable_interop=True)
# interop torch and lc
torch_tensor = torch.rand(16, dtype=torch.float32, device=device)
tensor_pointer = torch_tensor.data_ptr()
default_stream = torch.cuda.default_stream()
interop_buffer.interop_copy_from(tensor_pointer, default_stream.cuda_stream)

interop_buffer.copy_to(arr)
print('Printing torch tensor from lc')
for i in arr:
    print(i)
arr = np.ones(16, dtype=np.float32)
interop_buffer.copy_from(arr)
interop_buffer.interop_copy_to(tensor_pointer, default_stream.cuda_stream)
result_arr = torch_tensor.cpu().numpy()

print('Printing lc tensor from torch')
for i in result_arr:
    print(i)