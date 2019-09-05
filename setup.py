from setuptools import setup, find_packages

import cusignal

shortdesc = "GPU Accelerated Signal Processing Library"
longdesc = """
cuSignal builds on the RAPIDS, Chainer, Numba, and PyTorch ecosystem to yield easy to use, GPU accelerated signal processing functionals. Ported directly from scipy.signal, cuSignal tries to remain API compliant -- often via a direct copy.
"""

setup(
    name='cusignal',
    version='0.1',
    description=shortdesc,
    long_description=longdesc,
    url='https://github.com/awthomp/cuSignal',
    packages=find_packages(),
    author='Adam Thompson',
    include_package_data=True
)

