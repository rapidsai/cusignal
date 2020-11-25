# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

import cupy as cp
import os

from pathlib import Path


def _get_numSM():

    device_id = cp.cuda.Device()

    return device_id.attributes["MultiProcessorCount"]


def _get_max_smem():

    device_id = cp.cuda.Device()

    return device_id.attributes["MaxSharedMemoryPerBlock"]


def _get_max_tpb():

    device_id = cp.cuda.Device()

    return device_id.attributes["MaxThreadsPerBlock"]


def _get_tpb_bpg():

    numSM = _get_numSM()
    threadsperblock = 512
    blockspergrid = numSM * 20

    return threadsperblock, blockspergrid


def _get_function(fatbin, func):
    dir = os.path.dirname(Path(__file__).parent)

    module = cp.RawModule(
        path=dir + fatbin,
    )
    return module.get_function(func)


def _print_atts(func):
    if os.environ.get("CUSIGNAL_DEV_DEBUG") == "True":
        print("name:", func.kernel.name)
        print("max_threads_per_block:", func.kernel.max_threads_per_block)
        print("num_regs:", func.kernel.num_regs)
        print(
            "max_dynamic_shared_size_bytes:",
            func.kernel.max_dynamic_shared_size_bytes,
        )
        print("shared_size_bytes:", func.kernel.shared_size_bytes)
        print(
            "preferred_shared_memory_carveout:",
            func.kernel.preferred_shared_memory_carveout,
        )
        print("const_size_bytes:", func.kernel.const_size_bytes)
        print("local_size_bytes:", func.kernel.local_size_bytes)
        print("ptx_version:", func.kernel.ptx_version)
        print("binary_version:", func.kernel.binary_version)
        print()
