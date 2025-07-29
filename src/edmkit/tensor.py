# ruff: noqa: F401
import platform

# workaround
if platform.system() == "Darwin" and platform.processor() != "arm":
    import os

    os.environ["METAL_XCODE"] = "1"
    os.environ["DISABLE_COMPILER_CACHE"] = "1"

    print("Running on macOS, setting METAL_XCODE=1 and DISABLE_COMPILER_CACHE=1")


from tinygrad import Tensor
from tinygrad.dtype import dtypes
