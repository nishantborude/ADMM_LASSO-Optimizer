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

    def update(self):
        self.x = np.linalg.inv(self.A.dot(self.A.T) + self.rho).dot(self.A.dot(self.b) + self.rho * self.z - self.nu)
        self.z = self.x + self.nu / self.rho - (self.lamb / self.rho) * np.sign(self.z)
        self.nu = self.nu + self.rho * (self.x - self.z)

    def getparam(self):
        return self.x

    def get_diff(self):
        print(self.x - self.z)

def obj_func(A, x, b, lamb):
    return 0.5 * np.linalg.norm(A.T.dot(x) - b) ** 2 + lamb * np.linalg.norm(x, 1)
