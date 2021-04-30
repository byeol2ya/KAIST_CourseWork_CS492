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


# #A=(2,5)
# Z = torch.randn(5)
# A = torch.reshape(Z, (1,1,1,5))
# B = torch.randn(2,3,4,5)
# C = A*B
# D = torch.zeros(2,3,4,5)
#
# for i in range(D.shape[0]):
#     print('/n/n')
#     for j in range(D.shape[1]):
#         for k in range(D.shape[2]):
#             for l in range(D.shape[3]):
#                 D[i,j,k,l] = Z[l] * B[i,j,k,l]
#                 print(B[i,j,k,l],Z[l], D[i,j,k,l], C[i,j,k,l])
#                 if C[i,j,k,l] != D[i,j,k,l]:
#                     print('dif')
#
# #
# # dim = [100, 50, 30,15]
# # conv = nn.ModuleList([nn.Sequential(nn.Linear(dim[i], dim[i+1]),nn.BatchNorm1d(dim[i+1])) for i in range(len(dim)-1)])
# # print(conv)
# print('test')

# a = torch.tensor([[[1.0,  2.0], [3.0, 4.0], [5.0,  6.0]],[[1.0,  2.0], [3.0, 4.0], [5.0,  6.0]]])
# b = torch.tensor([[[8.0, 7.0], [9.0, 10.0], [11.0, 12.0]],[[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]])
# c = torch.cdist(a, b, p=2)
#
# print(c)

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # GPU에서 실행하려면 이 주석을 제거하세요.

# N은 배치 크기이며, D_in은 입력의 차원입니다;
# H는 은닉층의 차원이며, D_out은 출력 차원입니다.
N, D_in, H, D_out = 64, 1000, 100, 10

# 입력과 출력을 저장하기 위해 무작위 값을 갖는 Tensor를 생성합니다.
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# 가중치를 저장하기 위해 무작위 값을 갖는 Tensor를 생성합니다.
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # 사용자 정의 Function을 적용하기 위해 Function.apply 메소드를 사용합니다.
    # 여기에 'relu'라는 이름을 붙였습니다.
    relu = MyReLU.apply

    # 순전파 단계: Tensor 연산을 사용하여 예상되는 y 값을 계산합니다;
    # 사용자 정의 autograd 연산을 사용하여 ReLU를 계산합니다.
    y_pred = relu(x.mm(w1)).mm(w2)

    # 손실을 계산하고 출력합니다.
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())

    # autograde를 사용하여 역전파 단계를 계산합니다.
    loss.backward()

    # 경사하강법(gradient descent)을 사용하여 가중치를 갱신합니다.
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # 가중치 갱신 후에는 수동으로 변화도를 0으로 만듭니다.
        w1.grad.zero_()
        w2.grad.zero_()