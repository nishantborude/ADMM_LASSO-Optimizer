import numpy as np

class ADMM(object):
    """
    min f(x) + g(z)
    s.t. Ax + Bz = c
    """
    def __init__(self, A, b, lamb, rho):
        self.A = A
        self.b = b
        self.lamb = lamb
        self.rho = rho
        self.x = np.zeros((A.shape[0], 1))
        self.z = np.zeros((A.shape[0], 1))
        self.nu = np.zeros((A.shape[0], 1))

    def optimize(self):
        pos_grad =  1 * (self.z > 0)
        neg_grad = -1 * (self.z < 0)
        sub_grad = pos_grad + neg_grad
        self.z = self.x + self.nu / self.rho - (self.lamb / self.rho) * sub_grad

    def update(self):
        self.x = np.linalg.inv(self.A.dot(self.A.T) + self.rho).dot(self.A.dot(self.b) + self.rho * self.z - self.nu)
        self.optimize()
        self.nu = self.nu + self.rho * (self.x - self.z)

    def getparam(self):
        return self.x

def obj_func(A, x, b, lamb):
    return 0.5 * np.linalg.norm(A.T.dot(x) - b) ** 2 + lamb * np.linalg.norm(x, 1)
