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

import io
import versioneer
from setuptools import setup, find_packages
import sys
from os.path import dirname, join


SETUP_REQUIRES = ['setuptools >= 24.2.0']
SETUP_REQUIRES += ['wheel'] if 'bdist_wheel' in sys.argv else []


def read(*names, **kwargs):
    with io.open(
        join(dirname(__file__), *names),
        encoding=kwargs.get('encoding', 'utf8')
    ) as fh:
        return fh.read()


opts = dict(
    name='cusignal',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license="Apache 2.0",
    description="cuSignal - GPU Signal Processing",
    url="https://github.com/rapidsai/cusignal",
    author="NVIDIA Corporation",
    packages=find_packages(include=["cusignal", "cusignal.*"]),
    package_data={"": ["*.fatbin"]},
    include_package_data=True,
    zip_safe=False,
    project_urls={
        'Documentation': 'https://docs.rapids.ai/api/cusignal/stable/',
        'Issue Tracker': 'https://github.com/rapidsai/cusignal/issues',
    },
    python_requires='>=3.6',
    platforms=['manylinux2014_x86_64'],
    setup_requires=SETUP_REQUIRES,
    install_requires=[
        'numpy', 'numba', 'scipy',
    ],
)

if __name__ == '__main__':
    setup(**opts)
