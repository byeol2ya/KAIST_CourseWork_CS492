import torch
import torch.nn as nn
import numpy as np

# A, B, C, D = 3, 3, 2, 2
# c = torch.ones(A, B) * 2
# v = torch.randn(A, B, C, D)
#
# print(c.shape, v.shape)
# d = c[:, :, None, None] * v
# print((d[0, 0] == v[0, 0]* 2).all())
#
# nv = torch.ones(2, 3, 4)
# for i in range(2):
#     for j in range(3):
#         for k in range(4):
#             nv[i][j][k] = pow(10,i)*(j+k)
#
# nv_flatten = torch.flatten(nv,start_dim=1)
#
# A=(2,5)
# Z = torch.randn(2, 5)
# A = torch.reshape(Z, (2,1,1,5))
# B = torch.randn(2,3,4,5)
# C = A*B
# D = torch.zeros(2,3,4,5)
#
# for i in range(D.shape[0]):
#     for j in range(D.shape[1]):
#         for k in range(D.shape[2]):
#             for l in range(D.shape[3]):
#                 D[i,j,k,l] = Z[i,l] * B[i,j,k,l]
#                 if C[i,j,k,l] != D[i,j,k,l]:
#                     print('dif')
#
# dim = [100, 50, 30,15]
# conv = nn.ModuleList([nn.Sequential(nn.Linear(dim[i], dim[i+1]),nn.BatchNorm1d(dim[i+1])) for i in range(len(dim)-1)])
# print(conv)
# print('test')


#A=(2,5)
Z = torch.randn(5)
A = torch.reshape(Z, (1,1,1,5))
B = torch.randn(2,3,4,5)
C = A*B
D = torch.zeros(2,3,4,5)

for i in range(D.shape[0]):
    print('/n/n')
    for j in range(D.shape[1]):
        for k in range(D.shape[2]):
            for l in range(D.shape[3]):
                D[i,j,k,l] = Z[l] * B[i,j,k,l]
                print(B[i,j,k,l],Z[l], D[i,j,k,l], C[i,j,k,l])
                if C[i,j,k,l] != D[i,j,k,l]:
                    print('dif')

#
# dim = [100, 50, 30,15]
# conv = nn.ModuleList([nn.Sequential(nn.Linear(dim[i], dim[i+1]),nn.BatchNorm1d(dim[i+1])) for i in range(len(dim)-1)])
# print(conv)
print('test')
