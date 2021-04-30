# import numpy as np
#
# from pymoo.algorithms.nsga2 import NSGA2
# from pymoo.model.problem import Problem
# from pymoo.optimize import minimize
# from pymoo.visualization.scatter import Scatter
#
#
# class MyProblem(Problem):
#
#     def __init__(self):
#         super().__init__(n_var=2,
#                          n_obj=2,
#                          n_constr=2,
#                          xl=np.array([-2, -2]),
#                          xu=np.array([2, 2]),
#                          elementwise_evaluation=True)
#
#     def _evaluate(self, x, out, *args, **kwargs):
#         # f1 = x[0] ** 2 + x[1] ** 2
#         f1 = self.test(x[0],x[1])
#         f2 = (x[0] - 1) ** 2 + x[1] ** 2
#
#         g1 = 2 * (x[0] - 0.1) * (x[0] - 0.9) / 0.18
#         g2 = - 20 * (x[0] - 0.4) * (x[0] - 0.6) / 4.8
#
#         out["F"] = []
#         out["F"] += [f1, f2]
#         out["G"] = [g1, g2]
#
#     def test(self,x1,x2):
#         f1 = np.sqrt((x1-x2)*(x1-x2))
#         return f1
#
# problem = MyProblem()
#
# algorithm = NSGA2(pop_size=100)
#
# res = minimize(problem,
#                algorithm,
#                ("n_gen", 100),
#                verbose=True,
#                seed=1)
#
# plot = Scatter()
# plot.add(res.F, color="red")
# plot.show()
# print(res.X[-1])
#
# import torch
#
# from torch.autograd import Variable
#
# a = torch.ones(2,2)
#
# print(a)
#
#
#
# #옆에 requires_grad=True는 a값이 필요하다라는 것
#
# #위의 tensor를 생성했는데 연산을 추적하기 위해 requires_grad=True 설정
#
# a = Variable(a, requires_grad=True)
#
# print(a)
#
#
#
# print("---a.data---")
#
# print(a.data)
#
# #아무런 연산을 진행하지 않아서 None
#
# print("---a.grad---")
#
# print(a.grad)
#
# print("---a.grad_fn---")
#
# print(a.grad_fn)
#
#
#
# #+연산
#
# b = a+2
#
# print(b)
#
#
#
# #b의 제곱 연산
#
# c = b ** 2
#
# print(c)
#
# d = c * a+ b
#
# print(d)
# out = d.sum()
#
# print(out)
#
#
#
# out.backward()
#
# print(a.grad)
# print(b.grad)
#
import torch
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable
# saved_weights = torch.tensor([0.1, 0.2, 0.3, 0.25])
# loaded_weights =  Variable(saved_weights, requires_grad=True)
# weights = loaded_weights ** 2  # some function
# print(weights)
# #tensor([-0.5503,  0.4926, -2.1158, -0.8303])
# # Now, start to record operations done to weights
# weights.requires_grad_()
# weights.retain_grad()
# #out = weights.pow(2).sum()
# # print(weights.shape)
# grad_output = torch.zeros(*weights.size())
# grad_output[0] = 1
# weights.backward(grad_output)
# print(saved_weights.grad)
# print(loaded_weights.grad)
# print(weights.grad)

a = torch.tensor([1.0, 2.0, 3.0], requires_grad = True)
b = torch.tensor([3.0, 4.0, 5.0], requires_grad = True)
c = torch.tensor([6.0, 7.0, 8.0], requires_grad = True)

y=3*a + 2*b*b + torch.log(c)
# gradients = torch.FloatTensor([0.1, 1.0, 0.0001])
gradients = torch.FloatTensor([1.0, 0.0, 0.0])
y.backward(gradients,retain_graph=True)
# y.backward()

print(a.grad) # tensor([3.0000e-01, 3.0000e+00, 3.0000e-04])
print(b.grad) # tensor([1.2000e+00, 1.6000e+01, 2.0000e-03])
print(c.grad) # tensor([1.6667e-02, 1.4286e-01, 1.2500e-05])



d = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad = True)
e = torch.tensor([[6.0, 7.0], [8.0, 9.0]], requires_grad = True)
E = torch.zeros(2,2,*d.size(), requires_grad=True)
z=3*d + 2*torch.matmul(d,e)
J = torch.zeros(2,2,*d.size())
# gradients = torch.FloatTensor([0.1, 1.0, 0.0001])
# y.backward()
for i in range(2):
    for j in range(2):
        zero_gradients(d)
        if i == 1:
            gradients = torch.FloatTensor([[1.0, 0.0], [0.0, 0.0]])
        else:
            gradients = torch.FloatTensor([[0.0, 0.0], [0.0, 1.0]])
        z.backward(gradients,retain_graph=True)
        J[i,j] = d.grad
print(d.grad) # tensor([3.0000e-01, 3.0000e+00, 3.0000e-04])
print(e.grad) # tensor([1.2000e+00, 1.6000e+01, 2.0000e-03])
Jk = (E+J).sum()
print(Jk)
Jk.backward()
print(E.grad)