import torch
from torch.autograd import Variable

x = Variable(torch.randn(2, 3), requires_grad=True)
y = Variable(torch.randn(2, 3))

criterion = torch.nn.MSELoss()
ans = criterion(x, y)
loss = torch.sqrt(ans)
loss.backward()
print(x.grad)


t = torch.FloatTensor([[1, 2], [30, 40]])
print(t)
print((t[0][0] + t[1][0])*0.5)
print(t.mean())
print(t.mean(dim=0))
print(t.mean(dim=1))
print(t.mean(dim=-1))


a = {'aa':123}
b = {123:'aaa'}
c = dict(a.items() | b.items())
print(c)


a = torch.tensor([[2.5,3.4,11.1],[1.1,9.3,2.1]],dtype=torch.float32)
c = torch.tensor([2,3],dtype=torch.float32)
print(c)
c = c.unsqueeze(1)
d = a*c
print(a)
print(c)
print(d)


GPU_NUM = 0 # 원하는 GPU 번호 입력
# device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')

a = torch.tensor([1134324.314242341342414],dtype=torch.float32)
c = torch.tensor([111111111111111111111.111111111111111111111111],dtype=torch.float32)
d = torch.tensor([1.11111111111111111111111111111111111111111111111111111],dtype=torch.float32)
e = torch.tensor([0.00000000000000000010000000000001],dtype=torch.float32)
_eps = torch.finfo(torch.float32).eps
b = a +_eps
print
print(f'{a}\n{_eps}\n{b}')
f = a/e
print(torch.eps)
g= str(e.cpu().numpy())
print("{}의 자리수".format(10**(len(g)-1)))