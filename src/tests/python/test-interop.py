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
BUFFER_SIZE = 16
arr = np.zeros(BUFFER_SIZE, dtype=np.float32)
interop_buffer = Buffer(BUFFER_SIZE, float, enable_interop=True)
# interop torch and lc
torch_tensor = torch.rand(BUFFER_SIZE, dtype=torch.float32, device=device)
tensor_pointer = torch_tensor.data_ptr()
default_stream = torch.cuda.default_stream()
interop_buffer.interop_copy_from(tensor_pointer, default_stream.cuda_stream)

interop_buffer.copy_to(arr)
print('Printing torch tensor from lc')
for i in arr:
    print(i)
@func
def kernel():
    id = dispatch_id().x
    interop_buffer.write(id, float(id) + 0.114)
kernel(dispatch_size=(BUFFER_SIZE,1,1))
interop_buffer.interop_copy_to(tensor_pointer, default_stream.cuda_stream)
result_arr = torch_tensor.cpu().numpy()

print('Printing lc tensor from torch')
for i in result_arr:
    print(i)