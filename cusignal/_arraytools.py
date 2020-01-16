import cupy as cp
from numba import cuda
import numpy as np


# Return shared memory array -- necessary for zero copy between CPU and GPU on
# NVIDIA embedded platforms
# Confirmed working with Numba 0.45 and NVIDIA TX2, Nano, and Xavier
def get_shared_array(data, strides=None, order='C', stream=0, portable=False,
                     wc=True):
    shape = data.shape
    dtype = data.dtype

    # Allocate mapped, shared memory in Numba
    shared_mem_array = cuda.mapped_array(shape, dtype=dtype, strides=strides,
                                         order=order, stream=stream,
                                         portable=portable, wc=wc)

    # Load data into array space
    shared_mem_array[:] = data

    return shared_mem_array


# Return shared memory array - similar to np.zeros
def get_shared_mem(shape, dtype=np.float32, strides=None, order='C', stream=0,
                   portable=False, wc=True):
    return cuda.mapped_array(shape, dtype=dtype, strides=strides, order=order,
                             stream=stream, portable=portable, wc=wc)


def axis_slice(a, start=None, stop=None, step=None, axis=-1):
    """Take a slice along axis 'axis' from 'a'.

    Parameters
    ----------
    a : numpy.ndarray
        The array to be sliced.
    start, stop, step : int or None
        The slice parameters.
    axis : int, optional
        The axis of `a` to be sliced.

    Examples
    --------
    >>> a = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> axis_slice(a, start=0, stop=1, axis=1)
    array([[1],
           [4],
           [7]])
    >>> axis_slice(a, start=1, axis=0)
    array([[4, 5, 6],
           [7, 8, 9]])

    Notes
    -----
    The keyword arguments start, stop and step are used by calling
    slice(start, stop, step).  This implies axis_slice() does not
    handle its arguments the exactly the same as indexing.  To select
    a single index k, for example, use
        axis_slice(a, start=k, stop=k+1)
    In this case, the length of the axis 'axis' in the result will
    be 1; the trivial dimension is not removed. (Use numpy.squeeze()
    to remove trivial axes.)
    """
    a = cp.asarray(a)
    a_slice = [slice(None)] * a.ndim
    a_slice[axis] = slice(start, stop, step)
    b = a[tuple(a_slice)]
    return b


def axis_reverse(a, axis=-1):
    """Reverse the 1-d slices of `a` along axis `axis`.

    Returns axis_slice(a, step=-1, axis=axis).
    """
    return axis_slice(a, step=-1, axis=axis)


def odd_ext(x, n, axis=-1):
    """
    Odd extension at the boundaries of an array

    Generate a new ndarray by making an odd extension of `x` along an axis.

    Parameters
    ----------
    x : ndarray
        The array to be extended.
    n : int
        The number of elements by which to extend `x` at each end of the axis.
    axis : int, optional
        The axis along which to extend `x`.  Default is -1.

    Examples
    --------
    >>> from scipy.signal._arraytools import odd_ext
    >>> a = np.array([[1, 2, 3, 4, 5], [0, 1, 4, 9, 16]])
    >>> odd_ext(a, 2)
    array([[-1,  0,  1,  2,  3,  4,  5,  6,  7],
           [-4, -1,  0,  1,  4,  9, 16, 23, 28]])

    Odd extension is a "180 degree rotation" at the endpoints of the original
    array:

    >>> t = np.linspace(0, 1.5, 100)
    >>> a = 0.9 * np.sin(2 * np.pi * t**2)
    >>> b = odd_ext(a, 40)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(arange(-40, 140), b, 'b', lw=1, label='odd extension')
    >>> plt.plot(arange(100), a, 'r', lw=2, label='original')
    >>> plt.legend(loc='best')
    >>> plt.show()
    """
    x = cp.asarray(x)
    if n < 1:
        return x
    if n > x.shape[axis] - 1:
        raise ValueError(("The extension length n (%d) is too big. " +
                         "It must not exceed x.shape[axis]-1, which is %d.")
                         % (n, x.shape[axis] - 1))
    left_end = axis_slice(x, start=0, stop=1, axis=axis)
    left_ext = axis_slice(x, start=n, stop=0, step=-1, axis=axis)
    right_end = axis_slice(x, start=-1, axis=axis)
    right_ext = axis_slice(x, start=-2, stop=-(n + 2), step=-1, axis=axis)
    ext = cp.concatenate((2 * left_end - left_ext,
                          x,
                          2 * right_end - right_ext),
                         axis=axis)
    return ext


def even_ext(x, n, axis=-1):
    """
    Even extension at the boundaries of an array

    Generate a new ndarray by making an even extension of `x` along an axis.

    Parameters
    ----------
    x : ndarray
        The array to be extended.
    n : int
        The number of elements by which to extend `x` at each end of the axis.
    axis : int, optional
        The axis along which to extend `x`.  Default is -1.

    Examples
    --------
    >>> from scipy.signal._arraytools import even_ext
    >>> a = np.array([[1, 2, 3, 4, 5], [0, 1, 4, 9, 16]])
    >>> even_ext(a, 2)
    array([[ 3,  2,  1,  2,  3,  4,  5,  4,  3],
           [ 4,  1,  0,  1,  4,  9, 16,  9,  4]])

    Even extension is a "mirror image" at the boundaries of the original array:

    >>> t = np.linspace(0, 1.5, 100)
    >>> a = 0.9 * np.sin(2 * np.pi * t**2)
    >>> b = even_ext(a, 40)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(arange(-40, 140), b, 'b', lw=1, label='even extension')
    >>> plt.plot(arange(100), a, 'r', lw=2, label='original')
    >>> plt.legend(loc='best')
    >>> plt.show()
    """
    x = cp.asarray(x)
    if n < 1:
        return x
    if n > x.shape[axis] - 1:
        raise ValueError(("The extension length n (%d) is too big. " +
                         "It must not exceed x.shape[axis]-1, which is %d.")
                         % (n, x.shape[axis] - 1))
    left_ext = axis_slice(x, start=n, stop=0, step=-1, axis=axis)
    right_ext = axis_slice(x, start=-2, stop=-(n + 2), step=-1, axis=axis)
    ext = cp.concatenate((left_ext,
                          x,
                          right_ext),
                         axis=axis)
    return ext


def const_ext(x, n, axis=-1):
    """
    Constant extension at the boundaries of an array

    Generate a new ndarray that is a constant extension of `x` along an axis.

    The extension repeats the values at the first and last element of
    the axis.

    Parameters
    ----------
    x : ndarray
        The array to be extended.
    n : int
        The number of elements by which to extend `x` at each end of the axis.
    axis : int, optional
        The axis along which to extend `x`.  Default is -1.

    Examples
    --------
    >>> from scipy.signal._arraytools import const_ext
    >>> a = np.array([[1, 2, 3, 4, 5], [0, 1, 4, 9, 16]])
    >>> const_ext(a, 2)
    array([[ 1,  1,  1,  2,  3,  4,  5,  5,  5],
           [ 0,  0,  0,  1,  4,  9, 16, 16, 16]])

    Constant extension continues with the same values as the endpoints of the
    array:

    >>> t = np.linspace(0, 1.5, 100)
    >>> a = 0.9 * np.sin(2 * np.pi * t**2)
    >>> b = const_ext(a, 40)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(arange(-40, 140), b, 'b', lw=1, label='constant extension')
    >>> plt.plot(arange(100), a, 'r', lw=2, label='original')
    >>> plt.legend(loc='best')
    >>> plt.show()
    """
    x = cp.asarray(x)
    if n < 1:
        return x
    left_end = axis_slice(x, start=0, stop=1, axis=axis)
    ones_shape = [1] * x.ndim
    ones_shape[axis] = n
    ones = cp.ones(ones_shape, dtype=x.dtype)
    left_ext = ones * left_end
    right_end = axis_slice(x, start=-1, axis=axis)
    right_ext = ones * right_end
    ext = cp.concatenate((left_ext,
                          x,
                          right_ext),
                         axis=axis)
    return ext


def zero_ext(x, n, axis=-1):
    """
    Zero padding at the boundaries of an array

    Generate a new ndarray that is a zero padded extension of `x` along
    an axis.

    Parameters
    ----------
    x : ndarray
        The array to be extended.
    n : int
        The number of elements by which to extend `x` at each end of the
        axis.
    axis : int, optional
        The axis along which to extend `x`.  Default is -1.

    Examples
    --------
    >>> from scipy.signal._arraytools import zero_ext
    >>> a = np.array([[1, 2, 3, 4, 5], [0, 1, 4, 9, 16]])
    >>> zero_ext(a, 2)
    array([[ 0,  0,  1,  2,  3,  4,  5,  0,  0],
           [ 0,  0,  0,  1,  4,  9, 16,  0,  0]])
    """
    x = cp.asarray(x)
    if n < 1:
        return x
    zeros_shape = list(x.shape)
    zeros_shape[axis] = n
    zeros = cp.zeros(zeros_shape, dtype=x.dtype)
    ext = cp.concatenate((zeros, x, zeros), axis=axis)
    return ext


def as_strided(x, shape=None, strides=None):
    """
    Create a view into the array with the given shape and strides.
    .. warning:: This function has to be used with extreme care, see notes.
    Parameters
    ----------
    x : ndarray
        Array to create a new.
    shape : sequence of int, optional
        The shape of the new array. Defaults to ``x.shape``.
    strides : sequence of int, optional
        The strides of the new array. Defaults to ``x.strides``.
    Returns
    -------
    view : ndarray
    See also
    --------
    numpy.lib.stride_tricks.as_strided
    reshape : reshape an array.
    Notes
    -----
    ``as_strided`` creates a view into the array given the exact strides
    and shape. This means it manipulates the internal data structure of
    ndarray and, if done incorrectly, the array elements can point to
    invalid memory and can corrupt results or crash your program.
    """
    shape = x.shape if shape is None else tuple(shape)
    strides = x.strides if strides is None else tuple(strides)

    return cp.ndarray(shape=shape, dtype=x.dtype,
                      memptr=x.data, strides=strides)
