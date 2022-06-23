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
        super(PolyphaseDiff, self).__init__()
        self.up = up
        self.down = down
        self.filter_coeffs = filter_coeffs

    def forward(self, x):
        return FuncPolyphase.apply(x, self.filter_coeffs, self.up,
                                   self.down)
