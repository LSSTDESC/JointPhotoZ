import numpy as np
from scipy.stats import multivariate_normal

#['hess_traf', 'res', 'pi_trafo', 'hessian_laplace']

class LikeTF(object):
    def __init__(self, deconv_tf, midpoints):
        self.midpoints = midpoints
        self.mean_s = np.array(deconv_tf['pi_trafo'])
        self.hess = np.array(deconv_tf['h_new'])

    def loglike(self, nz_s):
        return 0.5*(self.mean_s - nz_s).dot(self.hess.dot(self.mean_s - nz_s))




class ListLikeTF(object):
    def __init__(self, list_deconv_tf, midpoints):
        self.midpoints = midpoints
        self.list_deconv_tf = list_deconv_tf
        self.cov_res = self.list_deconv_tf['cov_res'][0]
        self.hess_res = -np.linalg.inv(self.cov_res)
        self.mean_res = self.list_deconv_tf['mean_res'][0]

    def loglike(self, nz_s):
        return 0.5*(self.mean_res - nz_s).dot(self.hess_res.dot(self.mean_res - nz_s))




