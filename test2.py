import torch
import numpy as np

A, B, C, D = 3, 3, 2, 2
c = torch.ones(A, B) * 2
v = torch.randn(A, B, C, D)

print(c.shape, v.shape)
d = c[:, :, None, None] * v
print((d[0, 0] == v[0, 0]* 2).all())

nv = torch.ones(2, 3, 4)
for i in range(2):
    for j in range(3):
        for k in range(4):
            nv[i][j][k] = pow(10,i)*(j+k)

nv_flatten = torch.flatten(nv,start_dim=1)
print('test')