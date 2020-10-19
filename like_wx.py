from scipy.stats import norm
from scipy.stats import multivariate_normal
# from matplotlib import pyplot as plt
import numpy as np
import pickle
from scipy.interpolate import InterpolatedUnivariateSpline as interp
from scipy.integrate import simps


class GeneratorBiasModel(object):
    def __init__(self, model_biases, z_grid):
        self.model_low = model_biases['model_low']
        self.model_mid = model_biases['model_mid']
        self.model_high= model_biases['model_high']
        self.list_range = model_biases['list_range']
        self.z_grid = z_grid

    def get_model_low(self):
        return self.model_low.coef

    def get_model_mid(self):
        return self.model_mid.coef

    def get_model_high(self):
        return self.model_high.coef

    def set_model_low(self, coefs):
        self.model_low.coef = coefs

    def set_model_mid(self, coefs):
        self.model_mid.coef = coefs

    def set_model_high(self, coefs):
        self.model_high.coef = coefs

    def get_bias(self, z):
        list_result = np.zeros((len(z),))
        for idx, el_z in enumerate(z):
            if el_z < self.list_range[0]:
                list_result[idx] = self.model_low(el_z)
            elif (el_z >= self.list_range[0]) & (el_z < self.list_range[1]):
                list_result[idx] = self.model_mid(el_z)
            elif el_z > self.list_range[1]:
                list_result[idx] = self.model_high(el_z)

        return list_result



class RatioDist(object):

    def __init__(self, mean_1, std_1, mean_2, std_2):
        self.mean_1 = mean_1
        self.std_1 = std_1
        self.mean_2 = mean_2
        self.std_2 = std_2
        self.c = (self.mean_1/self.std_1)**2 + (self.mean_2/self.std_2)**2

    def b(self, z):
        return self.mean_1/self.std_1**2 * z + self.mean_2/self.std_2**2

    def a(self, z):
        return np.sqrt((z/self.std_1)**2 + (1/self.std_2)**2 )

    def d(self, z):
        return np.exp((self.b(z)**2 - self.c * self.a(z)**2)/(2 * self.a(z)**2))

    def p1alpha(self, z):
        return self.b(z) * self.d(z)/(self.a(z)**3)

    def p1beta(self, z):
        return norm.cdf(self.b(z)/self.a(z)) - norm.cdf(-self.b(z)/self.a(z))

    def pdf(self, z):
        prefac =  self.p1alpha(z) * 1/(np.sqrt(2 * np.pi) * self.std_1 * self.std_2)
        add_term = 1./(self.a(z)**2 * np.pi * self.std_1 * self.std_2) * np.exp(-self.c/2.)
        diff_cdf = self.p1beta(z)

        return prefac * diff_cdf + add_term

    def logpdf(self, z):
        #print(self.pdf(z))
        return np.log(self.pdf(z)+np.finfo(float).tiny)



class ProdRatioDist(object):
    def __init__(self, wx_sp, wx_sp_err, wx_ss, wx_ss_err):
        self.list_RatioDist = [RatioDist(wx_sp[i], wx_sp_err[i], wx_ss[i], wx_ss_err[i])
                               for i in range(len(wx_sp))]

    def get_loglike(self, vec):
        list_prod = np.array([self.list_RatioDist[i].logpdf(vec[i]) for i in range(len(self.list_RatioDist))])
        return np.sum(list_prod)


#data_wx[:, 0], wrb_mid_new, rb_err, ref_corr, ref_corr_err,
#ratio_wx, phot_nz, spec_nz

class LogLikeWX(object):
    def __init__(self, output_model, data_wx, mod=1):
        self.bias_model = GeneratorBiasModel(output_model, data_wx[:, 0])
        self.midpoints = data_wx[:, 0]
        self.bias_orig = self.bias_model.get_bias(self.midpoints)
        self.wx_sp = data_wx[:, 1]
        self.wx_sp_err = data_wx[:, 2]/mod
        self.wx_ss = data_wx[:, 3]
        self.wx_ss_err = data_wx[:, 4]/mod
        self.prodRatio = ProdRatioDist(self.wx_sp, self.wx_sp_err, self.wx_ss, self.wx_ss_err)
        self.spec_nz = data_wx[:, 7]/np.trapz(data_wx[:, 7], self.midpoints)
        self.phot_nz = data_wx[:, 6]/np.trapz(data_wx[:, 6], self.midpoints)


    def log(self, nz, bratio):
        bias_low = self.bias_model.get_model_low()
        bias_mid = self.bias_model.get_model_mid()
        bias_high = self.bias_model.get_model_high()

        bias_low[0] = bratio[0]
        bias_mid[0] = bratio[1]
        bias_high[0] = bratio[2]

        bias_low[1] = bratio[3]
        bias_mid[1] = bratio[4]
        bias_high[1] = bratio[5]

        bias_low[2] = bratio[6]
        bias_mid[2] = bratio[7]
        bias_high[2] = bratio[8]

        self.bias_model.set_model_low(bias_low)
        self.bias_model.set_model_mid(bias_mid)
        self.bias_model.set_model_high(bias_high)

        bias_fnkt = self.bias_model.get_bias(self.midpoints)
        nz = nz/np.trapz(nz, self.midpoints)

        vec = bias_fnkt * nz/self.spec_nz
        #vec_orig = self.bias_orig *self.phot_nz/self.spec_nz

        # print(vec)
        return self.prodRatio.get_loglike(vec)



