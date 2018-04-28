import numpy as np
from admm import ADMM, obj_func

n = 1000
d = 20

A = np.random.randn(d, n)
b = np.random.randn(n, 1)
x = np.random.randn(d, 1)

lamb = 0.01
rho = 1
MAX_ITER = 10

print("Initial Value of the function: ",  obj_func(A, x, b, lamb))

optim = ADMM(A, b, lamb, rho, parallel=True)
# print(obj_func(A, x, b, lamb))
for i in range(MAX_ITER):
    optim.update()
    x = optim.getparam()
    print(obj_func(A, x, b, lamb))

print("Optmial Value of the function: ",  obj_func(A, x, b, lamb))
# print("Optimal Parameter: ",  x)
# print(optim.get_diff())
