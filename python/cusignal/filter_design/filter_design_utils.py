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


def _validate_sos(sos):
    """Helper to validate a SOS input"""
    sos = cp.atleast_2d(sos)
    if sos.ndim != 2:
        raise ValueError("sos array must be 2D")
    n_sections, m = sos.shape
    if m != 6:
        raise ValueError("sos array must be shape (n_sections, 6)")
    if not (sos[:, 3] == 1).all():
        raise ValueError("sos[:, 3] should be all ones")
    return sos, n_sections
