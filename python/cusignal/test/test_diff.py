import torch
from cusignal.diff import 


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
    module = PolyphaseDiff(up, down, filter_coeffs)
    kwargs = {"eps": eps}
    if rtol > 0:
        kwargs["rtol"] = rtol
    else:
        kwargs["atol"] = atol
    gradcheck(module, inputs, **kwargs, raise_exception = True)


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
    module = PolyphaseDiff(up, down, window)
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
