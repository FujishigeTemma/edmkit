import gc
import sys
from collections.abc import Generator

import pytest


@pytest.fixture(autouse=True)
def free_device_cache() -> Generator[None]:
    """Release the buffers tinygrad keeps for reuse."""
    yield
    if "tinygrad" in sys.modules:
        from tinygrad import Device
        from tinygrad.device import LRUAllocator

        gc.collect()
        if isinstance(allocator := Device[Device.DEFAULT].allocator, LRUAllocator):
            allocator.free_cache()
