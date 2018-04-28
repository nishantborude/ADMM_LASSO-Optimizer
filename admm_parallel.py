import numpy as np
from joblib import Parallel, delayed
import multiprocessing


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
        self.x = np.zeros((A.shape[1], A.shape[0]))
        self.z = np.zeros((A.shape[0], 1))
        self.nu = np.zeros((A.shape[1], A.shape[0]))
        self.x_bar = np.mean(self.x, 0).reshape(-1 , 1)
        self.nu_bar = np.mean(self.nu, 0).reshape(-1 , 1)
        self.num_cores = multiprocessing.cpu_count()

    def update(self):
        for i in range(self.A.shape[1]):
            self.x[i] = (self.A[:, i] * self.b[i] + self.rho * self.z.reshape(-1) - self.nu[i]) / (self.A[:, i].dot(self.A[:, i].T) + self.rho)
        self.x_bar = np.mean(self.x, 0).reshape(-1 , 1)
        self.z = (self.x_bar + self.nu_bar / self.rho) - (self.lamb / self.rho) * np.sign(self.z)
        for i in range(self.A.shape[1]):
            self.nu[i] = self.nu[i] + (self.rho * (self.x[i] - self.z.reshape(-1)))
        self.nu_bar = np.mean(self.nu, 0).reshape(-1 , 1)

    def getparam(self):
        return self.x_bar

    def get_diff(self):
        print(self.x_bar - self.z)

def obj_func(A, x, b, lamb):
    return 0.5 * np.linalg.norm(A.T.dot(x) - b) ** 2 + lamb * np.linalg.norm(x, 1)
