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

from string import Template


# Custom Cupy raw kernel implementing upsample, filter, downsample operation
# Matthew Nicely - mnicely@nvidia.com
_cupy_lfilter_src = Template(
    """
$header

extern "C" {
    __global__ void _cupy_lfilter(
            const int x_len,
            const int a_len,
            const ${datatype} * __restrict__ x,
            const ${datatype} * __restrict__ a,
            const ${datatype} * __restrict__ b,
            ${datatype} * __restrict__ out) {

        for ( int tid = 0; tid < x_len; tid++) {

            ${datatype} isw {};
            ${datatype} wos {};

            // Create input_signal_windows
            if( tid > ( a_len ) ) {
                for ( int i = 0; i < a_len; i++ ) {
                    isw += x[tid - i] * b[i];
                    wos += out[tid - i] * a[i];
                }
            } else {
                for ( int i = 0; i <= tid; i++ ) {
                    isw += x[tid - i] * b[i];
                    wos += out[tid - i] * a[i];
                }
            }

            isw -= wos;

            out[tid] = isw / a[0];
        }
    }
}
"""
)
