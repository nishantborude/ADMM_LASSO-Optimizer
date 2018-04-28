import numpy as np
from joblib import Parallel, delayed
import multiprocessing


class ADMM(object):
    """
    min f(x) + g(z)
    s.t. Ax + Bz = c
    """
    def __init__(self, A, b, lamb, rho, parallel=False):
        self.A = A
        self.b = b
        self.lamb = lamb
        self.rho = rho
        self.z = np.zeros((A.shape[0], 1))
        self.parallel = parallel
        if self.parallel:
            self.x = np.zeros((A.shape[1], A.shape[0]))
            self.nu = np.zeros((A.shape[1], A.shape[0]))
            self.x_bar = np.mean(self.x, 0).reshape(-1 , 1)
            self.nu_bar = np.mean(self.nu, 0).reshape(-1 , 1)
            self.num_cores = multiprocessing.cpu_count()
        else:
            self.x = np.zeros((A.shape[0], 1))
            self.nu = np.zeros((A.shape[0], 1))

    def update(self):
        if self.parallel:
            self.update_parallel()
        else:
            self.update_serial()

    def getparam(self):
        if self.parallel:
            return self.x_bar
        return self.x

    def get_diff(self):
        if self.parallel:
            print(self.x_bar - self.z)
        else:
            print(self.x - self.z)

    def update_one_x(self, i):
        self.x[i] = (self.A[:, i] * self.b[i] + self.rho * self.z.reshape(-1) - self.nu[i]) / (self.A[:, i].dot(self.A[:, i].T) + self.rho)

    def update_one_nu(self, i):
        self.nu[i] = self.nu[i] + (self.rho * (self.x[i] - self.z.reshape(-1)))

    def update_parallel(self):
        Parallel(n_jobs=self.num_cores)(delayed(self.update_one_x)(i) for i in range(self.A.shape[1]))
        self.x_bar = np.mean(self.x, 0).reshape(-1 , 1)
        self.z = (self.x_bar + self.nu_bar / self.rho) - (self.lamb / self.rho) * np.sign(self.z)
        Parallel(n_jobs=self.num_cores)(delayed(self.update_one_nu)(i) for i in range(self.A.shape[1]))
        self.nu_bar = np.mean(self.nu, 0).reshape(-1 , 1)

    def update_serial(self):
        self.x = np.linalg.inv(self.A.dot(self.A.T) + self.rho).dot(self.A.dot(self.b) + self.rho * self.z - self.nu)
        self.z = self.x + self.nu / self.rho - (self.lamb / self.rho) * np.sign(self.z)
        self.nu = self.nu + self.rho * (self.x - self.z)

def obj_func(A, x, b, lamb):
    return 0.5 * np.linalg.norm(A.T.dot(x) - b) ** 2 + lamb * np.linalg.norm(x, 1)
