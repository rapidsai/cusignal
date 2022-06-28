import torch
import pytest
from numpy import allclose
from cusignal.diff import ResamplePoly
from cusignal import resample_poly
from torch.autograd.gradcheck import gradcheck


@pytest.mark.parametrize("device", ['cpu', 'cuda'])
@pytest.mark.parametrize("up", [1, 10])
@pytest.mark.parametrize("down", [1, 7, 10, 13])
@pytest.mark.parametrize("filter_size", [1, 10])
def test_gradcheck(device, up, down, filter_size,
                   eps=1e-3, atol=1e-1, rtol=-1):
    '''
    Verifies that our backward method works.
    '''
    up = torch.Tensor([up])
    down = torch.Tensor([down])
    filter_coeffs = torch.randn(filter_size, requires_grad = True,
                                dtype = torch.double,
                                device = device)
    inputs = torch.randn(100, dtype = torch.double, requires_grad = True,
                         device = device)
    module = ResamplePoly(up, down, filter_coeffs)
    kwargs = {"eps": eps}
    if rtol > 0:
        kwargs["rtol"] = rtol
    else:
        kwargs["atol"] = atol
    gradcheck(module, inputs, **kwargs, raise_exception = True)


@pytest.mark.parametrize("device", ['cpu', 'cuda'])
@pytest.mark.parametrize("x_size", [30, 100])
@pytest.mark.parametrize("filter_size", [5, 20])
@pytest.mark.parametrize("up", [1, 10])
@pytest.mark.parametrize("down", [1, 7, 10])
def test_forward(device, x_size, up, down, filter_size):
    '''
    Verifies that our module agress with scipy's implementation
    on randomly generated examples.

    gpupath = True accepts cupy typed windows.
    gpupath = False accepts numpy types windows.
    '''
    device = torch.device(device)
    gpupath = True
    if device != 'cuda':
        gpupath = False
    x = torch.randn(x_size, device = device)
    up = torch.Tensor([up])
    down = torch.Tensor([down])
    window = torch.randn(filter_size, device = device)
    # The module requires a torch tensor window
    module = ResamplePoly(up, down, window)
    # resample_poly requires a cupy or numpy array window
    window = window.cpu().numpy()
    if gpupath:
        window = cp.array(window)
    bench_resample = resample_poly(x, up, down, window = window,
                                   gpupath = gpupath)
    our_resample = module.forward(x)
    if not allclose(bench_resample, our_resample, atol=1e-4):
        raise Exception("Module does not agree with resample")
