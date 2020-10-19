import numpy as np
from matplotlib import pyplot as plt
import json
from scipy.stats import multivariate_normal
from scipy.interpolate import InterpolatedUnivariateSpline
import pickle


def transformation_logit(pi_vec):
    return np.log(pi_vec[:-1]/(pi_vec[-1]+np.finfo(float).tiny) + np.finfo(float).tiny)

def backtransform_logit(s_vec):
    denom = 1. + np.sum(np.exp(s_vec))

    vec_pi = np.exp(s_vec)/denom
    vec_pi = vec_pi.tolist()
    vec_pi.append(1/denom)
    return np.array(vec_pi)


data = np.loadtxt('output_regularization_models_31_bins_500k_0_08.pickle4mod_chain1_trace_loglike_bratio_second_DESI_version2_largechain.dat')
data_wx = np.loadtxt('new_cross_correlation_measurements_analytic_bias.dat')


# with open('output_regularization_models_50_25_bins_5000k_0_0001.pickle', 'r') as infile:
#     deconvolution_res = pickle.load(infile)

# cov_res = deconvolution_res['cov_res'][0]
# hess_res = -np.linalg.inv(cov_res)
# mean_res = deconvolution_res['mean_res'][0]

# samples = np.random.multivariate_normal(mean_res, cov_res, size=10000)

# plt.plot(samples[:, -2], samples[:, -1])
# plt.show()

# # plt.plot(data[:, 0], data[:, 1], '.')
# # plt.show()
# # 1/0
# pi_vec = np.array([backtransform_logit(el) for el in data])
# s_vec = np.array([transformation_logit(el) for el in pi_vec])
# # plt.plot(s_vec[:, -2], s_vec[:, -1], '.')
# # plt.show()
# pi_vec = np.array([el/np.trapz(el, data_wx[:, 0]) for el in pi_vec])

# # plt.plot(data[:, -1], '.', alpha=0.2)
# # plt.show()

# pi_vec = np.array([backtransform_logit(el) for el in data])


# for i in range(len(data)):
plt.plot(data, '.', alpha=0.2)
   # print(np.float(len(np.unique(pi_vec[:, i])))/np.float(len(pi_vec[:, i])))
plt.show()
# list_post_mean = []
# for el in pi_vec:
#     list_post_mean.append(np.trapz(el*data_wx[:, 0], data_wx[:, 0]))

# plt.hist(list_post_mean, 100)
# plt.show()
# plt.hist(list_post_mean, 100)
# plt.axvline(np.trapz(data_wx[:, -2] * data_wx[:, 0], data_wx[:, 0]))

# # plt.plot(data_wx[:, 0], np.median(pi_vec, axis=0)/np.sum(np.median(pi_vec, axis=0)))
# # plt.plot(data_wx[:, 0], data_wx[:, -2]/np.sum(data_wx[:, -2]))
# plt.show()
