from torch import tensor, device, allclose

from optim import MFAC

torch_device = device('cuda:0')
parameters = [
    tensor([[1., 2., 3.], [4., 5., 6.]], device=torch_device),
    tensor([[9., 4.], [7., 3.], [8., 1.]], device=torch_device)
]

optimizer = MFAC(parameters, moddev=torch_device, optdev=torch_device)
parameters[0].grad = tensor([[4., 2., 6.], [3., 9., 1.]], device=torch_device)
parameters[1].grad = tensor([[1., 1.], [2., 7.], [4., 5.]], device=torch_device)
optimizer.step()

expected_params = [
    tensor([[0.9832, 1.9916, 2.9748],
            [3.9874, 4.9621, 5.9958]], device=torch_device),
    tensor([[8.9958, 3.9958],
            [6.9916, 2.9705],
            [7.9832, 0.9789]], device=torch_device)]

assert allclose(parameters[0], expected_params[0], rtol=1e-03, atol=1e-03)
assert allclose(parameters[1], expected_params[1], rtol=1e-03, atol=1e-03)

print(parameters[0])
print(parameters[1])