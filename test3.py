import torch

def get_tensor_info(tensor):
  info = []
  for name in ['requires_grad', 'is_leaf', 'retains_grad', 'grad_fn', 'grad']:
    info.append(f'{name}({getattr(tensor, name, None)})')
  info.append(f'tensor({str(tensor)})')
  return ' '.join(info)

x = torch.tensor([5.0,3.0], requires_grad=True)
y = x ** 3
z = (torch.log(y)).sum()
k = x[:]
l = (k ** 2).sum()

print('x', get_tensor_info(x))
print('y', get_tensor_info(y))
print('z', get_tensor_info(z))
print('k', get_tensor_info(k))

l.backward(retain_graph=True)

print('x_after_backward', get_tensor_info(x))
print('y_after_backward', get_tensor_info(y))
print('z_after_backward', get_tensor_info(z))
print('k_after_backward', get_tensor_info(k))

z.backward()