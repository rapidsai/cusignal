# Copyright (c) 2022, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

import cusignal
import cupy as cp

@pytest.mark.parametrize("dtype", [cp.ubyte, cp.complex64])
@pytest.mark.parametrize("shape", [1024, (32, 32)])
def test_get_pinned_mem(dtype, shape):
    arr = cusignal.get_pinned_mem(shape=shape, dtype=dtype)

    if isinstance(shape, int):
        shape = (shape, )
    
    assert arr.shape == shape
    assert arr.dtype == dtype
