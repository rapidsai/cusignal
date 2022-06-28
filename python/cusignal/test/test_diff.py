import pytest
from numpy import sqrt
from numpy import allclose
from cusignal import resample_poly


try:
    import torch
    from cusignal.diff import ResamplePoly
    from torch.autograd.gradcheck import gradcheck
except ImportError:
    pytest.skip(f"skipping pytorch dependant tests in {__file__}",
                allow_module_level=True)


@pytest.mark.parametrize("device", ['cpu', 'cuda'])
@pytest.mark.parametrize("up", [1, 3, 10])
@pytest.mark.parametrize("down", [1, 7, 10])
@pytest.mark.parametrize("filter_size", [1, 10])
def test_gradcheck(device, up, down, filter_size,
                   eps=1e-3, atol=1e-1, rtol=-1):
    '''
    Verifies that our backward method works.
    '''
    '''
    up = torch.Tensor([up])
    down = torch.Tensor([down])
    '''
    filter_coeffs = torch.randn(filter_size, requires_grad=True,
                                dtype=torch.double,
                                device=device)
    inputs = torch.randn(100, dtype=torch.double, requires_grad=True,
                         device=device)
    module = ResamplePoly(up, down, filter_coeffs)
    kwargs = {"eps": eps}
    if rtol > 0:
        kwargs["rtol"] = rtol
    else:
        kwargs["atol"] = atol
    gradcheck(module, inputs, **kwargs, raise_exception=True)


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
    x = torch.randn(x_size, device=device)
    '''
    up = torch.Tensor([up])
    down = torch.Tensor([down])
    '''
    window = torch.randn(filter_size, device=device)
    # The module requires a torch tensor window
    module = ResamplePoly(up, down, window)
    # resample_poly requires a cupy or numpy array window
    window = window.cpu().numpy()
    if gpupath:
        window = cp.array(window)
    bench_resample = resample_poly(x, up, down, window=window,
                                   gpupath=gpupath)
    our_resample = module.forward(x)
    if not allclose(bench_resample, our_resample.detach().cpu(), atol=1e-4):
        raise Exception("Module does not agree with resample")


def test_backprop(device='cuda', iters=100, filter_size=10, up=7,
                  down=3, x_size=1000):
    '''
    Demonstration of how to use ResamplePoly with back prop
    '''
    x = torch.linspace(-sqrt(10), sqrt(10), x_size, device='cuda')
    y = torch.sin(x)
    window = torch.randn(filter_size, device=device)
    model = torch.nn.Sequential(
        ResamplePoly(up, down, window),
        torch.nn.Linear(in_features=ResamplePoly.output_size(x_size, up, down),
                        out_features=1000).to(device)
    )
    loss_fn = torch.nn.MSELoss(reduction='sum')
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    loss.backward()
