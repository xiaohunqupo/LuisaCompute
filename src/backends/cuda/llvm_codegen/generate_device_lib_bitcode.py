# This script generates the device library bitcode for CUDA backend using LLVM.

from shutil import which
from pathlib import Path

def embed_bitcode(input_file: str, output_file: str, c_array_name: str):
    pass


def find_nvvm_libdevice_path() -> str:
    nvcc_path = which("nvcc")
    assert nvcc_path is not None
    nvvm_libdevice_path = Path(nvcc_path).parent.parent / "nvvm" / "libdevice"
    print(nvvm_libdevice_path)
    return nvvm_libdevice_path

def main():
    nvvm_libdevice_path = find_nvvm_libdevice_path()
    print(nvvm_libdevice_path)

