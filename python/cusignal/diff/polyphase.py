import torch
import numpy as np
import cupy as cp
from torch import flip
from torch.autograd import Function
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.nn.init import uniform_
from torch.autograd.gradcheck import gradcheck
from math import gcd
from cusignal import resample_poly
from cusignal import choose_conv_method
from cusignal import correlate


class FuncPolyphase(Function):
    @staticmethod
    def get_start_index(length):
        if length <= 2:
            return 0
        return (length - 1) // 2
    
    @staticmethod
    def best_corr(sig1, sig2, mode):
        method = choose_conv_method(sig1, sig2, mode = mode)
        out = correlate(sig1, sig2, mode = mode, method = method)
        return out

    @staticmethod
    def forward(ctx, x, filter_coeffs, up, down):
        device        = x.device.type
        x             = x.detach()
        filter_coeffs = filter_coeffs.detach()
        up            = up.detach()
        down          = down.detach()

        x_size    = x.shape[0]
        filt_size = filter_coeffs.shape[0]
        up        = int(up[0])
        down      = int(down[0])
        ud_gcd    = gcd(up, down)
        up        = up // ud_gcd
        down      = down // ud_gcd

        if (up == 1 and down == 1):
            x_out = x
            inverse_size = torch.Tensor([x.shape[0]])
            out_len = torch.Tensor([0])
            x_up = None
        else:
            if 'cuda' in device:
                gpupath = True
                window  = cp.array(filter_coeffs)
            else:
                gpupath = False
                window  = filter_coeffs.numpy()
            x_out = resample_poly(x, up, down, window = window,
                                  gpupath = gpupath)
            inverse_size = up * x_size + filt_size - 1
            x_up = torch.zeros(up * x_size, device = device, dtype = x.dtype)
            x_up[::up] = up * x

        ctx.save_for_backward(torch.Tensor([x_size]), filter_coeffs,
                              torch.Tensor([up]),
                              torch.Tensor([down]),
                              torch.Tensor([inverse_size]),
                              torch.Tensor([len(x_out)]), x_up)
        return(torch.Tensor(cp.asnumpy(x_out)))

    @staticmethod
    def backward(ctx, gradient):
        gradient = gradient.detach()
        x_size, filter_coeffs, up, down, inverse_size, out_len, x_up \
        = ctx.saved_tensors

        device        = gradient.device.type
        x_size        = int(x_size[0])
        gradient_size = gradient.shape[0]
        filt_size     = filter_coeffs.shape[0]
        up            = int(up[0])
        down          = int(down[0])
        start         = FuncPolyphase.get_start_index(filt_size)
        inverse_size  = int(inverse_size)
        out_x_len     = int(out_len)
        filter_coeffs = filter_coeffs.type(gradient.dtype)

        if (up == 1 and down == 1):
            # J_x up \times J_x conv
            out_x = gradient
            # J_f conv
            out_f = torch.zeros(filter_coeffs.shape[0],
                                device = device,
                                dtype = filter_coeffs.dtype)
        else:
            tmp = torch.zeros(out_x_len, device = device,
                              dtype = gradient.dtype)
            tmp[:gradient.shape[0]] = gradient
            gradient = tmp
            gradient_up = torch.zeros(inverse_size,
                                      device = device,
                                      dtype = gradient.dtype)
            extra = bool((inverse_size - start) % down)
            tmp = torch.zeros((inverse_size - start) // down + extra,
                              device = device,
                              dtype = filter_coeffs.dtype)
            tmp[:gradient.shape[0]] = gradient
            gradient_up[start :: down] = torch.clone(tmp)

            out_x = FuncPolyphase.best_corr(gradient_up,
                                            filter_coeffs,
                                            mode = 'valid')
            out_x = up * out_x[::up]
            out_f = FuncPolyphase.best_corr(gradient_up, x_up,
                                            mode = 'valid')
        out_x = torch.as_tensor(out_x[:x_size],
                                device = device)
        out_f = torch.as_tensor(out_f[:filter_coeffs.shape[0]],
                                device = device)
        return(out_x, out_f, None, None)


class PolyphaseDiff(Module):
    def __init__(self, up, down, filter_coeffs):
        super(Resample, self).__init__()
        self.up = up
        self.down = down
        self.filter_coeffs = filter_coeffs

    def forward(self, x):
        return FuncPolyphase.apply(x, self.filter_coeffs, self.up,
                                   self.down)


def accept_reps(f):
    def wrapper(repetitions = 1, **kwargs):
        for i in range(repetitions):
            f(**kwargs)
    return wrapper


@accept_reps
def gradcheck_main(eps=1e-6, atol=1e-3, rtol=-1, device = 'cpu'):
    '''
    Verifies that our backward method works.
    '''
    up = torch.randint(1, 20, (1,), requires_grad = False)
    down = torch.randint(1, 20, (1,), requires_grad = False)
    filter_size = np.random.randint(10,30)
    filter_coeffs = torch.randn(filter_size, requires_grad = True,
                                dtype = torch.double,
                                device = device)
    inputs = torch.randn(100, dtype = torch.double, requires_grad = True,
                         device = device)
    module = Resample(up, down, filter_coeffs)
    kwargs = {"eps": eps}
    if rtol > 0:
        kwargs["rtol"] = rtol
    else:
        kwargs["atol"] = atol
    gradcheck(module, inputs, **kwargs, raise_exception = True)


@accept_reps
def forward_main(gpupath = True):
    '''
    Verifies that our module agress with scipy's implementation
    on randomly generated examples.

    gpupath = True accepts cupy typed windows.
    gpupath = False accepts numpy types windows.
    '''
    if gpupath:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    x_size = np.random.randint(30, 100)
    filter_size = np.random.randint(5, 20)
    x = torch.randn(x_size, device = device)
    up = torch.randint(1, 20, (1,), device = device)
    down = torch.randint(1, 20, (1,), device = device)
    window = torch.randn(filter_size, device = device)
    # The module requires a torch tensor window
    module = Resample(up, down, window)
    # resample_poly requires a cupy or numpy array window
    window = window.cpu().numpy()
    if gpupath:
        window = cp.array(window)
    bench_resample = resample_poly(x, up, down, window = window,
                                   gpupath = gpupath)
    our_resample = module.forward(x)
    if not np.allclose(bench_resample, our_resample, atol=1e-4):
        print(f"up: {up}, down: {down}")
        print(f"scipy result: {scipy_resample[:10]}")
        print(f"our result: {our_resample[:10]}")
        raise Exception("Forward main failure")


if __name__ == '__main__':
    #forward_main(100)
    gradcheck_main(100, eps = 1e-3, atol = 1e-1)
    print("tests complete")
