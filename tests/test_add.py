import torch

torch.ops.load_library("./build/libtest.so")

a = torch.ones(100).float().cuda()
b = a.clone()
c = torch.zeros(100).float().cuda()


print(torch.ops.test_ops.add(a, b, c))
