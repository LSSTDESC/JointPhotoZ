import numpy as np
from matplotlib import pyplot as plt
import pickle
import json
import pandas as pd
import pyccl as ccl
from box_stats import *
from scipy.interpolate import InterpolatedUnivariateSpline


def transformation_logit(pi_vec):
    return np.log(pi_vec[:-1]/(pi_vec[-1]+np.finfo(float).tiny) + np.finfo(float).tiny)

def backtransform_logit(s_vec):
    denom = 1. + np.sum(np.exp(s_vec))

    vec_pi = np.exp(s_vec)/denom
    vec_pi = vec_pi.tolist()
    vec_pi.append(1/denom)
    return np.array(vec_pi)

data_subsample = pd.read_hdf('../DES_DNF_WX_binning_zspec.fits', key='zspec')

data_wx = np.loadtxt('../new_cross_correlation_measurements_analytic_bias.dat')

# with open('output_regularization_models_25_bins_5000k_0_0001.pickle', 'rb') as infile:
#     data_25_bins_5000k_0_0001 = pickle.load(infile)


with open('output_regularization_models_50_25_bins_5000k_0_0001.pickle', 'rb') as infile:
    data_50_25_bins_5000k_0_0001 = pickle.load(infile)


with open('output_regularization_models_25_25_bins_5000k_0_0001.pickle', 'rb') as infile:
    data_25_25_bins_5000k_0_0001 = pickle.load(infile)

with open('output_regularization_models_50_25_bins_5000k_10.pickle', 'rb') as infile:
    data_50_25_bins_5000k_10 = pickle.load(infile)



# print(np.percentile(np.mean(data_50_25_bins_5000k_10['result_list_merged'][0][-1]), 99))
# print(0.8767296684961651)

# 1/0
with open('output_regularization_models_50_25_bins_5000k_1.pickle', 'rb') as infile:
    data_50_25_bins_5000k_1 = pickle.load(infile)


with open('output_regularization_models_31_bins_500k_0_08.pickle', 'rb') as infile:
    data_31_bins_500k_0_08 = pickle.load(infile)

# with open('output_regularization_models_31_bins_50k_0_08.pickle', 'rb') as infile:
#     data_31_bins_50k_0_08 = pickle.load(infile)

with open('output_regularization_models_50_bins_50k_0_1.pickle', 'rb') as infile:
    data_50_bins_50k_0_1 = pickle.load(infile)


def transformation_logit(pi_vec):
    return np.log(pi_vec[:-1]/(pi_vec[-1]+np.finfo(float).tiny) + np.finfo(float).tiny)

def backtransform_logit(s_vec):
    denom = 1. + np.sum(np.exp(s_vec))

    vec_pi = np.exp(s_vec)/denom
    vec_pi = vec_pi.tolist()
    vec_pi.append(1/denom)
    return np.array(vec_pi)


data = np.loadtxt('output_regularization_models_50_25_bins_5000k_0_0001.pickle4mod_chain1_trace_vec_s_test_large_std_bratio_second_DESI.dat')

pi_vec = np.array([backtransform_logit(el) for el in data])
pi_vec = np.array([el/np.trapz(el, data_wx[:, 0]) for el in pi_vec])
mean_pi_vec = np.array([np.trapz(el*data_wx[:, 0], data_wx[:, 0]) for el in pi_vec])

# result_list_merged_50_25_bins_5000k_0_0001 = data_50_25_bins_5000k_0_0001['result_list_merged']
# result_list_merged_25_25_bins_5000k_0_0001 = data_25_25_bins_5000k_0_0001['result_list_merged']
# result_list_merged_31_bins_500k_0_08 = data_31_bins_500k_0_08['result_list_merged']


# list_data_50_25_bins_5000k_0_0001 = []
# list_data_25_25_bins_5000k_0_0001 = []
# list_data_31_bins_500k_0_08 = []

# print(data_50_25_bins_5000k_0_0001['result_list_merged'][0][-1])
# for idx, el in enumerate(list_data_50_25_bins_5000k_0_0001):
#     print(el[2])
#     list_data_50_25_bins_5000k_0_0001.append(el[2])

# for idx, el in enumerate(list_data_25_25_bins_5000k_0_0001):
#     list_data_25_25_bins_5000k_0_0001.append(el[2])

# for idx, el in enumerate(list_data_31_bins_500k_0_08):
#     list_data_31_bins_500k_0_08.append(el[2])


# list_data_50_25_bins_5000k_0_0001 = np.array(list_data_50_25_bins_5000k_0_0001)
# list_data_25_25_bins_5000k_0_0001 = np.array(list_data_25_25_bins_5000k_0_0001)
# list_data_31_bins_500k_0_08 = np.array(list_data_31_bins_500k_0_08)

# print(list_data_31_bins_500k_0_08)

# plt.figure(1)

# print(len(list_data_31_bins_500k_0_08))
# print(len(list_data_50_25_bins_5000k_0_0001))
# print(len(list_data_25_25_bins_5000k_0_0001))

# Calculate the angular cross-spectrum of the two tracers as a function of ell


# print(data_50_25_bins_5000k_0_0001['result_list_merged'][0][1])
# print(data_50_25_bins_5000k_0_0001['result_list_merged'][0][0][0])

# plt.plot(data_50_25_bins_5000k_0_0001['result_list_merged'][0][0][0],
#     np.median(data_50_25_bins_5000k_0_0001['result_list_merged'][0][1], axis=0))
# plt.show()



# Calculate the angular cross-spectrum of the two tracers as a function of ell
ell = np.arange(60, 1000, 1)
cosmo = ccl.Cosmology(
    Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
    transfer_function='bbks')

data_wx = np.loadtxt('../new_cross_correlation_measurements_analytic_bias.dat')
print(np.trapz(data_wx[:, -2], data_wx[:, 0]))
plt.plot(data_wx[:, 0], data_wx[:, -2])
plt.show()
lenskernel_true = ccl.NumberCountsTracer(cosmo, has_rsd=False, dndz=(data_wx[:, 0], data_wx[:, -2]), bias=(data_wx[:, 0], np.repeat(1.0, len(data_wx[:, 0]))))

true_cl = np.mean(ccl.angular_cl(cosmo, lenskernel_true, lenskernel_true, ell))
print(true_cl)


# plt.figure(50)

cl_mat_50_25_bins_5000k_0_0001 = []
for idx in range(0, data_50_25_bins_5000k_0_0001['result_list_merged'][0][1].shape[0], 30):
    print(idx)
    lenskernel = ccl.NumberCountsTracer(cosmo,has_rsd=False,bias=(data_wx[:, 0], np.repeat(1.0, len(data_wx[:, 0]))),
        dndz=(data_50_25_bins_5000k_0_0001['result_list_merged'][0][0][0],
              data_50_25_bins_5000k_0_0001['result_list_merged'][0][1][idx]/np.trapz(data_50_25_bins_5000k_0_0001['result_list_merged'][0][1][idx], data_50_25_bins_5000k_0_0001['result_list_merged'][0][0][0])))

    cl_mat_50_25_bins_5000k_0_0001.append(np.mean(ccl.angular_cl(cosmo, lenskernel, lenskernel, ell)))

cl_mat_50_25_bins_5000k_0_0001 = np.array(cl_mat_50_25_bins_5000k_0_0001)


cl_mat_50_25_bins_5000k_0_0001_wx = []
for idx in range(0, pi_vec.shape[0], 30):
    print(idx)
    lenskernel = ccl.NumberCountsTracer(cosmo,has_rsd=False, bias=(data_wx[:, 0], np.repeat(1.0, len(data_wx[:, 0]))),
        dndz=(data_wx[:, 0], pi_vec[idx]))
    cl_mat_50_25_bins_5000k_0_0001_wx.append(np.mean(ccl.angular_cl(cosmo, lenskernel, lenskernel, ell)))
    # plt.plot(ell, (ccl.angular_cl(cosmo, lenskernel, lenskernel, ell) - true_cl)/true_cl)

cl_mat_50_25_bins_5000k_0_0001_wx = np.array(cl_mat_50_25_bins_5000k_0_0001_wx)
# plt.xscale('log')
# plt.yscale('log')
# plt.show()


cl_mat_25_25_bins_5000k_0_0001 = []
for idx in range(0, data_25_25_bins_5000k_0_0001['result_list_merged'][0][1].shape[0], 30):
    print(idx)
    lenskernel = ccl.NumberCountsTracer(cosmo,has_rsd=False, bias=(data_wx[:, 0], np.repeat(1.0, len(data_wx[:, 0]))),
        dndz=(data_25_25_bins_5000k_0_0001['result_list_merged'][0][0][0],
              data_25_25_bins_5000k_0_0001['result_list_merged'][0][1][idx]/np.trapz(data_25_25_bins_5000k_0_0001['result_list_merged'][0][1][idx], data_25_25_bins_5000k_0_0001['result_list_merged'][0][0][0])))
    cl_mat_25_25_bins_5000k_0_0001.append(np.mean(ccl.angular_cl(cosmo, lenskernel, lenskernel, ell)))
cl_mat_25_25_bins_5000k_0_0001 = np.array(cl_mat_25_25_bins_5000k_0_0001)


cl_mat_50_25_bins_5000k_10 = []
for idx in range(0, data_50_25_bins_5000k_10['result_list_merged'][0][1].shape[0], 30):
    print(idx)
    lenskernel = ccl.NumberCountsTracer(cosmo,has_rsd=False, bias=(data_wx[:, 0], np.repeat(1.0, len(data_wx[:, 0]))),
        dndz=(data_50_25_bins_5000k_10['result_list_merged'][0][0][0],
              data_50_25_bins_5000k_10['result_list_merged'][0][1][idx]/np.trapz(data_50_25_bins_5000k_10['result_list_merged'][0][1][idx], data_50_25_bins_5000k_10['result_list_merged'][0][0][0])))
    cl_mat_50_25_bins_5000k_10.append(np.mean(ccl.angular_cl(cosmo, lenskernel, lenskernel, ell)))
cl_mat_50_25_bins_5000k_10 = np.array(cl_mat_50_25_bins_5000k_10)


cl_mat_50_25_bins_5000k_1 = []
for idx in range(0, data_50_25_bins_5000k_1['result_list_merged'][0][1].shape[0], 30):
    print(idx)
    lenskernel = ccl.NumberCountsTracer(cosmo,has_rsd=False,bias=(data_wx[:, 0], np.repeat(1.0, len(data_wx[:, 0]))),
        dndz=(data_50_25_bins_5000k_1['result_list_merged'][0][0][0],
              data_50_25_bins_5000k_1['result_list_merged'][0][1][idx]/np.trapz(data_50_25_bins_5000k_1['result_list_merged'][0][1][idx], data_50_25_bins_5000k_1['result_list_merged'][0][0][0])))
    cl_mat_50_25_bins_5000k_1.append(np.mean(ccl.angular_cl(cosmo, lenskernel, lenskernel, ell)))
cl_mat_50_25_bins_5000k_1 = np.array(cl_mat_50_25_bins_5000k_1)


print(np.median(cl_mat_50_25_bins_5000k_0_0001))
print(np.percentile(cl_mat_50_25_bins_5000k_0_0001, 10))
print(np.percentile(cl_mat_50_25_bins_5000k_0_0001, 90))

print(np.median(cl_mat_25_25_bins_5000k_0_0001))
print(np.percentile(cl_mat_25_25_bins_5000k_0_0001, 10))
print(np.percentile(cl_mat_25_25_bins_5000k_0_0001, 90))


sig_e = 0.23
n = 5.7e8
fsky = 0.436

sig_cl = np.sqrt(2/((2*ell + 1)*fsky))*(true_cl + sig_e**2/(2*n))
sig_mean_cl = 1/np.float(len(sig_cl)) * np.sqrt(np.sum(sig_cl**2))



print(sig_mean_cl)
plt.figure(0)

stats = {}
print(cl_mat_50_25_bins_5000k_0_0001)
# Compute the boxplot stats with our desired percentiles
stats['A'] = my_boxplot_stats((cl_mat_50_25_bins_5000k_0_0001 - true_cl)/true_cl, labels=['Tik. Regul. Low'],
  percents=[16, 84], whis=(2.5, 97.5))[0]
stats['B'] = my_boxplot_stats((cl_mat_50_25_bins_5000k_1 - true_cl)/true_cl,
    labels=['Tik. Regul.\nMedium'], percents=[16, 84], whis=(2.5, 97.5))[0]


stats['C'] = my_boxplot_stats((cl_mat_50_25_bins_5000k_10 - true_cl)/true_cl,
    labels=['Tik. Regul.\nHigh'], percents=[16, 84],whis=(2.5, 97.5))[0]
stats['D'] = my_boxplot_stats((cl_mat_25_25_bins_5000k_0_0001 - true_cl)/true_cl,
    labels=['Oversmoothing'], percents=[16, 84], whis=(2.5, 97.5))[0]
stats['E'] = my_boxplot_stats((cl_mat_50_25_bins_5000k_0_0001_wx - true_cl)/true_cl,
    labels=['Tik. Regul.\nLow + WX'], percents=[16, 84], whis=(2.5, 97.5))[0]
fig, ax = plt.subplots(1, 1)
# Plot boxplots from our computed statistics
bp = ax.bxp([stats['A'], stats['B'],
             stats['C'], stats['D'], stats['E']],
             positions=range(5), showfliers=False)

# Colour the lines in the boxplot blue
for element in bp.keys():
    plt.setp(bp[element], color='C3')

# plt.boxplot(np.column_stack(((cl_mat_50_25_bins_5000k_0_0001 - true_cl)/true_cl,
#     (cl_mat_25_25_bins_5000k_0_0001 - true_cl)/true_cl)), whis=(16, 84))

# plt.axhspan(-sig_mean_cl/true_cl, sig_mean_cl/true_cl, -1, 3, color='grey', alpha=0.2)
# plt.axhspan(-2*sig_mean_cl/true_cl, 2*sig_mean_cl/true_cl, -1, 3, color='grey', alpha=0.2)
plt.axhline(0, -1, 3, color='grey')
plt.ylabel(r'Relative Difference $(C_{\rm \ell}^{\rm Estim.} - C_{\rm \ell}^{\rm True})/C_{\rm \ell}^{\rm True}$',
  fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.title('Regularization LSS LSST SR N=5000k', fontsize=14)
plt.ylim([-0.025, 0.18])

plt.savefig("reg_tf_only_cl_lss.pdf",
               bbox_inches='tight',
               dpi=300,
               pad_inches=0)


plt.figure(1)
# data = np.column_stack((data_50_25_bins_5000k_0_0001['result_list_merged'][0][-1],
#                         data_50_25_bins_5000k_1['result_list_merged'][0][-1],
#                         data_50_25_bins_5000k_10['result_list_merged'][0][-1],
#                        data_25_25_bins_5000k_0_0001['result_list_merged'][0][-1],
#                        mean_pi_vec))

# , list_data_31_bins_50k_0_08[0],
#                         list_data_31_bins_50k_0_1[0]))

stats = {}
print(cl_mat_50_25_bins_5000k_0_0001)
# Compute the boxplot stats with our desired percentiles
stats['A'] = my_boxplot_stats(data_50_25_bins_5000k_0_0001['result_list_merged'][0][-1],
  labels=['Tik. Regul.\nLow'], percents=[16, 84], whis=(2.5, 97.5))[0]
stats['B'] = my_boxplot_stats(data_50_25_bins_5000k_1['result_list_merged'][0][-1],
    labels=['Tik. Regul.\nMedium'], percents=[16, 84], whis=(2.5, 97.5))[0]


stats['C'] = my_boxplot_stats(data_50_25_bins_5000k_10['result_list_merged'][0][-1],
    labels=['Tik. Regul.\nHigh'], percents=[16, 84], whis=(2.5, 97.5))[0]
stats['D'] = my_boxplot_stats(data_25_25_bins_5000k_0_0001['result_list_merged'][0][-1],
    labels=['Oversmoothing'], percents=[16, 84], whis=(2.5, 97.5))[0]
stats['E'] = my_boxplot_stats(mean_pi_vec,
    labels=['Tik. Regul.\nLow + WX'], percents=[16, 84], whis=(2.5, 97.5))[0]
fig, ax = plt.subplots(1, 1)
# Plot boxplots from our computed statistics
bp = ax.bxp([stats['A'], stats['B'],
             stats['C'], stats['D'], stats['E']],
             positions=range(5), showfliers=False)

# Colour the lines in the boxplot blue
for element in bp.keys():
    plt.setp(bp[element], color='C3')

# plt.boxplot(np.column_stack(((cl_mat_50_25_bins_5000k_0_0001 - true_cl)/true_cl,
#     (cl_mat_25_25_bins_5000k_0_0001 - true_cl)/true_cl)), whis=(16, 84))

lsst_sr_wl_1 = 0.002 * (1 + np.mean(data_subsample))[0]
lsst_sr_wl_10 = 0.001 * (1 + np.mean(data_subsample))[0]
lsst_sr_lss_1 = 0.005 * (1 + np.mean(data_subsample))[0]
lsst_sr_lss_10 = 0.003 * (1 + np.mean(data_subsample))[0]

print('discretization error and wl_1')
print(np.trapz(data_wx[:, -2]*data_wx[:, 0], data_wx[:, 0]) - np.mean(data_subsample))
print(lsst_sr_lss_1)
plt.axhline(np.trapz(data_wx[:, -2]*data_wx[:, 0], data_wx[:, 0]) - lsst_sr_wl_10, label='DESC SRD Y10 WL', color='black', alpha=0.4)
plt.axhline(np.trapz(data_wx[:, -2]*data_wx[:, 0], data_wx[:, 0]) + lsst_sr_wl_10, color='black', alpha=0.4)

plt.axhline(np.trapz(data_wx[:, -2]*data_wx[:, 0], data_wx[:, 0])- lsst_sr_wl_1, ls='--', label='DESC SRD Y1 WL', color='black', alpha=0.4)
plt.axhline(np.trapz(data_wx[:, -2]*data_wx[:, 0], data_wx[:, 0])+lsst_sr_wl_1, ls='--', color='black', alpha=0.4)


plt.axhline(np.trapz(data_wx[:, -2]*data_wx[:, 0], data_wx[:, 0])- lsst_sr_lss_1, ls='-.', label='DESC SRD Y1 LSS', color='black', alpha=0.4)
plt.axhline(np.trapz(data_wx[:, -2]*data_wx[:, 0], data_wx[:, 0])+lsst_sr_lss_1, ls='-.', color='black', alpha=0.4)

plt.axhline(np.trapz(data_wx[:, -2]*data_wx[:, 0], data_wx[:, 0])- lsst_sr_lss_10, ls=':', label='DESC SRD Y10 LSS', color='black', alpha=0.4)
plt.axhline(np.trapz(data_wx[:, -2]*data_wx[:, 0], data_wx[:, 0])+lsst_sr_lss_10, ls=':', color='black', alpha=0.4)

# plt.axhspan(np.trapz(data_wx[:, -2]*data_wx[:, 0], data_wx[:, 0])- lsst_sr_wl_10,
#   np.trapz(data_wx[:, -2]*data_wx[:, 0], data_wx[:, 0])+lsst_sr_wl_10, color='grey', alpha=0.2, label='SR Y10 WL')
# plt.axhspan(np.trapz(data_wx[:, -2]*data_wx[:, 0], data_wx[:, 0])- lsst_sr_wl_1,
#   np.trapz(data_wx[:, -2]*data_wx[:, 0], data_wx[:, 0])+lsst_sr_wl_1, color='cyan', alpha=0.2, label='SR Y1 WL')
# plt.axhspan(np.trapz(data_wx[:, -2]*data_wx[:, 0], data_wx[:, 0])- lsst_sr_lss_1,
#   np.trapz(data_wx[:, -2]*data_wx[:, 0], data_wx[:, 0])+lsst_sr_lss_1, color='magenta', alpha=0.2, label='SR Y1 LSS')
# plt.axhspan(np.trapz(data_wx[:, -2]*data_wx[:, 0], data_wx[:, 0])- lsst_sr_lss_10,
#   np.trapz(data_wx[:, -2]*data_wx[:, 0], data_wx[:, 0])+lsst_sr_lss_10, color='orange', alpha=0.2, label='SR Y10 LSS')

plt.axhline(np.trapz(data_wx[:, -2]*data_wx[:, 0], data_wx[:, 0]), alpha=0.4, color='black')
plt.ylabel(r'Posterior Mean $\left\langle z \right\rangle$', fontsize=14)
# plt.axhspan(np.trapz(data_wx[:, -2]*data_wx[:, 0], data_wx[:, 0]) - 0.007869,
#             np.trapz(data_wx[:, -2]*data_wx[:, 0], data_wx[:, 0]) + 0.007869, alpha=0.2, color='blue')
# plt.axhspan(np.trapz(data_wx[:, -2]*data_wx[:, 0], data_wx[:, 0])-2*0.01,
#   np.trapz(data_wx[:, -2]*data_wx[:, 0], data_wx[:, 0])+2*0.01, color='grey', alpha=0.2)
plt.legend(fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
#plt.title('Regularization Post Mean LSST SR N=5000k', fontsize=14)
plt.ylim([0.864, 0.94])
plt.savefig("reg_tf_only_mean.pdf",
               bbox_inches='tight',
               dpi=300,
               pad_inches=0)


result_list_merged_25_bins_5000k_0_0001 = data_25_25_bins_5000k_0_0001['result_list_merged']
result_list_merged_50_bins_50k_0_1 = data_50_bins_50k_0_1['result_list_merged']
result_list_merged_31_bins_500k_0_08 = data_31_bins_500k_0_08['result_list_merged']
# result_list_merged_31_bins_50k_0_08 = data_31_bins_50k_0_08['result_list_merged']
# result_list_merged_31_bins_50k_0_1 = data_31_bins_50k_0_1['result_list_merged']

#read in trace and convert:
#data_trace = np.loadtxt('output_regularization_models_31_bins_500k_0_08.pickle4mod_chain1_trace_vec_s_test_large_std_bratio_second_DESI.dat')
data_trace = np.loadtxt('output_regularization_models_31_bins_500k_0_08.pickle4mod_chain1_trace_vec_s_test_large_std_bratio_second_DESI_version2_largechain.dat')
pi_vec = np.array([backtransform_logit(el) for el in data_trace])
pi_vec = np.array([el/np.trapz(el, data_wx[:, 0]) for el in pi_vec])

plt.figure(20)

for idx, el in enumerate(result_list_merged_50_bins_50k_0_1):
    if idx==0:
        plt.fill_between(el[0][0], np.percentile(el[1], 16, axis=0),
            np.percentile(el[1], 84, axis=0), alpha=0.2, label='Small Sample Deconvolution', color='grey')
    else:
        plt.fill_between(el[0][0], np.percentile(el[1], 16, axis=0),
            np.percentile(el[1], 84, axis=0), alpha=0.2, color='grey')


for idx, el in enumerate(result_list_merged_31_bins_500k_0_08):
    if idx==0:
        plt.fill_between(el[0][0], np.percentile(el[1], 16, axis=0),
            np.percentile(el[1], 84, axis=0), alpha=0.6, label='Medium Sample (500k)', color='cyan')
    else:
        plt.fill_between(el[0][0], np.percentile(el[1], 16, axis=0),
            np.percentile(el[1], 84, axis=0), alpha=0.2, color='cyan')

for idx, el in enumerate(result_list_merged_25_bins_5000k_0_0001):
    if idx==0:
        plt.fill_between(el[0][0], np.percentile(el[1], 16, axis=0),
            np.percentile(el[1], 84, axis=0), alpha=0.8, label='Large Sample (5000k), Oversmoothing', color='magenta')
    else:
        plt.fill_between(el[0][0], np.percentile(el[1], 16, axis=0),
            np.percentile(el[1], 84, axis=0), alpha=0.2, color='magenta')

plt.fill_between(data_wx[:, 0], np.percentile(pi_vec, 16, axis=0), np.percentile(pi_vec, 84, axis=0), color='red', alpha=0.4,
                 label='Large Sample Joint Constraint')



plt.plot(data_wx[:, 0], data_wx[:, -2], '--', color='black', label='True Distribution')
plt.legend(fontsize=10)
plt.xlabel('Redshift z', fontsize=12)
plt.ylabel(r'Sample Photometric Redshift Distribution', fontsize=12)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylim([0, 1.4])
#plt.title('LSST SR, Photometric Likelihood', fontsize=14)
plt.savefig('lsst_sr_photometric_like4.pdf', bbox_inches='tight', dpi=300)

model = InterpolatedUnivariateSpline(data_wx[:, 0], data_wx[:, -2], ext=1, k=1)

plt.figure(3)


for idx, el in enumerate(result_list_merged_50_bins_50k_0_1):
    if idx==0:
        plt.fill_between(el[0][0], np.percentile((el[1] - model(el[0][0]))/model(el[0][0]), 16, axis=0),
            np.percentile((el[1] - model(el[0][0]))/model(el[0][0]), 84, axis=0), alpha=0.2, label='LSST SR, N=50k, T=0.1, MB=50', color='grey')
    else:
        plt.fill_between(el[0][0], np.percentile((el[1] - model(el[0][0]))/model(el[0][0]), 16, axis=0),
            np.percentile((el[1] - model(el[0][0]))/model(el[0][0]), 84, axis=0), alpha=0.2, color='grey')


for idx, el in enumerate(result_list_merged_31_bins_500k_0_08):
    if idx==0:
        plt.fill_between(el[0][0], np.percentile((el[1] - model(el[0][0]))/model(el[0][0]), 16, axis=0),
            np.percentile((el[1] - model(el[0][0]))/model(el[0][0]), 84, axis=0), alpha=0.6, label='LSST SR, N=500k, T=0.08, MB=31', color='cyan')
    else:
        plt.fill_between(el[0][0], np.percentile((el[1] - model(el[0][0]))/model(el[0][0]), 16, axis=0),
            np.percentile((el[1] - model(el[0][0]))/model(el[0][0]), 84, axis=0), alpha=0.2, color='cyan')

for idx, el in enumerate(result_list_merged_25_bins_5000k_0_0001):
    if idx==0:
        plt.fill_between(el[0][0], np.percentile((el[1] - model(el[0][0]))/model(el[0][0]), 16, axis=0),
            np.percentile((el[1] - model(el[0][0]))/model(el[0][0]), 84, axis=0), alpha=0.8, label='LSST SR, N=5000k, T=0.0001, MB=25', color='magenta')
    else:
        plt.fill_between(el[0][0], np.percentile((el[1] - model(el[0][0]))/model(el[0][0]), 16, axis=0),
            np.percentile((el[1] - model(el[0][0]))/model(el[0][0]), 84, axis=0), alpha=0.2, color='magenta')

plt.fill_between(data_wx[:, 0], np.percentile((pi_vec - data_wx[:, -2])/data_wx[:, -2], 16, axis=0),
                 np.percentile((pi_vec - data_wx[:, -2])/data_wx[:, -2], 84, axis=0), color='red', alpha=0.4,
                 label='WX, Full b(z), LSST SR, N=500k, T=0.08, MB=31')


plt.legend(fontsize=10, loc='upper left')

plt.plot(data_wx[:, 0], np.repeat(0, len(data_wx[:, 0])), '--', color='black')
plt.xlabel(r'Redshift $z$', fontsize=14)
plt.ylabel(r'$(p_{\rm post} - p_{\rm true})/p_{\rm true}$', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.title('LSST SR, Photometric Likelihood', fontsize=14)
plt.savefig('lsst_sr_photometric_like_relative_diff4.pdf', bbox_inches='tight', dpi=300)

# plt.figure(4)


# for idx, el in enumerate(result_list_merged_50_bins_50k_0_1):
#     if idx==0:
#         plt.plot(el[0][0], np.percentile((el[1] - model(el[0][0]))/model(el[0][0]), 16, axis=0),
#           alpha=0.2, label='Small Sample (50k)', color='grey')
#         plt.plot(el[0][0], np.percentile((el[1] - model(el[0][0]))/model(el[0][0]), 84, axis=0),
#           alpha=0.2, color='grey')
#         # plt.fill_between(el[0][0], np.percentile((el[1] - model(el[0][0]))/model(el[0][0]), 16, axis=0),
#         #     np.percentile((el[1] - model(el[0][0]))/model(el[0][0]), 84, axis=0), alpha=0.2, label='LSST SR, N=50k, T=0.1, MB=50', color='grey')
#     else:
#         plt.plot(el[0][0], np.percentile((el[1] - model(el[0][0]))/model(el[0][0]), 16, axis=0),
#           alpha=0.2, color='grey')
#         plt.plot(el[0][0], np.percentile((el[1] - model(el[0][0]))/model(el[0][0]), 84, axis=0),
#           alpha=0.2, color='grey')
#         # plt.fill_between(el[0][0], np.percentile((el[1] - model(el[0][0]))/model(el[0][0]), 16, axis=0),
#         #     np.percentile((el[1] - model(el[0][0]))/model(el[0][0]), 84, axis=0), alpha=0.2, color='grey')


# for idx, el in enumerate(result_list_merged_31_bins_500k_0_08):
#     if idx==0:
#         plt.plot(el[0][0], np.percentile((el[1] - model(el[0][0]))/model(el[0][0]), 16, axis=0),
#           alpha=0.6, label='Medium Sample (500k)', color='cyan')
#         plt.plot(el[0][0], np.percentile((el[1] - model(el[0][0]))/model(el[0][0]), 84, axis=0),
#           alpha=0.6, color='cyan')
#     #     plt.fill_between(el[0][0], np.percentile((el[1] - model(el[0][0]))/model(el[0][0]), 16, axis=0),
#     #         np.percentile((el[1] - model(el[0][0]))/model(el[0][0]), 84, axis=0), alpha=0.6, label='LSST SR, N=500k, T=0.08, MB=31', color='cyan')
#     else:
#         plt.plot(el[0][0], np.percentile((el[1] - model(el[0][0]))/model(el[0][0]), 16, axis=0),
#           alpha=0.6, color='cyan')
#         plt.plot(el[0][0], np.percentile((el[1] - model(el[0][0]))/model(el[0][0]), 84, axis=0),
#           alpha=0.6, color='cyan')
#         # plt.fill_between(el[0][0], np.percentile((el[1] - model(el[0][0]))/model(el[0][0]), 16, axis=0),
#         #     np.percentile((el[1] - model(el[0][0]))/model(el[0][0]), 84, axis=0), alpha=0.2, color='cyan')

# for idx, el in enumerate(result_list_merged_25_bins_5000k_0_0001):
#     if idx==0:
#         plt.plot(el[0][0], np.percentile((el[1] - model(el[0][0]))/model(el[0][0]), 16, axis=0), alpha=0.8, label='Large Sample (5000k), Oversmoothing', color='magenta')
#         plt.plot(el[0][0], np.percentile((el[1] - model(el[0][0]))/model(el[0][0]), 84, axis=0), alpha=0.8, color='magenta')
#         # plt.fill_between(el[0][0], np.percentile((el[1] - model(el[0][0]))/model(el[0][0]), 16, axis=0),
#         #     np.percentile((el[1] - model(el[0][0]))/model(el[0][0]), 84, axis=0), alpha=0.8, label='LSST SR, N=5000k, T=0.0001, MB=25', color='magenta')
#     else:
#         plt.plot(el[0][0], np.percentile((el[1] - model(el[0][0]))/model(el[0][0]), 16, axis=0), alpha=0.8, color='magenta')
#         plt.plot(el[0][0], np.percentile((el[1] - model(el[0][0]))/model(el[0][0]), 84, axis=0), alpha=0.8, color='magenta')


# # plt.fill_between(data_wx[:, 0], np.percentile((pi_vec - data_wx[:, -2])/data_wx[:, -2], 16, axis=0),
# #                  np.percentile((pi_vec - data_wx[:, -2])/data_wx[:, -2], 84, axis=0), color='red', alpha=0.4,
# #                  label='WX, Full b(z), LSST SR, N=500k, T=0.08, MB=31')
# plt.plot(data_wx[:, 0], np.percentile((pi_vec - data_wx[:, -2])/data_wx[:, -2], 16, axis=0), color='red', alpha=0.4,
#                  label='Medium Sample (500k), Fid. Setup & WX')
# plt.plot(data_wx[:, 0], np.percentile((pi_vec - data_wx[:, -2])/data_wx[:, -2], 84, axis=0), color='red', alpha=0.4)



# plt.legend(fontsize=10, loc='upper left')

# plt.plot(data_wx[:, 0], np.repeat(0, len(data_wx[:, 0])), '--', color='black')
# plt.xlabel(r'Redshift $z$', fontsize=14)
# plt.ylabel(r'$(n^{B}_{\rm post} - n^{B}_{\rm true})/n^{B}_{\rm true}$', fontsize=14)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)

# plt.title('LSST SR, Photometric Likelihood', fontsize=14)
# plt.savefig('lsst_sr_photometric_like_relative_diff.pdf', bbox_inches='tight', dpi=300)




# def set_axis_style(ax, labels):
#     ax.get_xaxis().set_tick_params(direction='out')
#     ax.xaxis.set_ticks_position('bottom')
#     ax.set_xticks(np.arange(1, len(labels) + 1))
#     ax.set_xticklabels(labels)
#     ax.set_xlim(0.25, len(labels) + 0.75)
#     ax.set_xlabel('Sample name')


# list_data_50_50k_0_1 = []
# list_data_31_500k_0_08 = []
# list_data_25_5000k_0_0001 = []
# list_data_31_bins_50k_0_08 = []
# list_data_31_bins_50k_0_1 = []

# for idx, el in enumerate(result_list_merged_50_bins_50k_0_1):
#     list_data_50_50k_0_1.append(el[2])

# for idx, el in enumerate(result_list_merged_31_bins_500k_0_08):
#     list_data_31_500k_0_08.append(el[2])

# for idx, el in enumerate(result_list_merged_25_bins_5000k_0_0001):
#     list_data_25_5000k_0_0001.append(el[2])

# for idx, el in enumerate(result_list_merged_31_bins_50k_0_08):
#     list_data_31_bins_50k_0_08.append(el[2])

# for idx, el in enumerate(result_list_merged_31_bins_50k_0_1):
#     list_data_31_bins_50k_0_1.append(el[2])

# list_data_50_50k_0_1 = np.array(list_data_50_50k_0_1)
# list_data_31_500k_0_08 = np.array(list_data_31_500k_0_08)
# list_data_25_5000k_0_0001 = np.array(list_data_25_5000k_0_0001)
# list_data_31_bins_50k_0_08 = np.array(list_data_31_bins_50k_0_08)
# list_data_31_bins_50k_0_1 = np.array(list_data_31_bins_50k_0_1)

# print(list_data_50_50k_0_1.shape)


# data = np.column_stack((list_data_50_50k_0_1[0], list_data_31_500k_0_08[0],
#                         list_data_25_5000k_0_0001[0], list_data_31_bins_50k_0_08[0],
#                         list_data_31_bins_50k_0_1[0]))


# plt.boxplot(data, labels=['LSST SR, N=500k, B=50, T=0.08, MB=31',
#                           'LSST SR, N=5000k, B=50, T=0.0001, MB=25', 'LSST SR, N=5000k, B=25, T=0.0001, MB=25'])

# plt.axhline(np.trapz(data_wx[:, -2]*data_wx[:, 0], data_wx[:, 0]), color='black')
# plt.ylim([0.870, 1.0])
# plt.show()
# plt.axhline(np.trapz(data_wx[:, -2]*data_wx[:, 0], data_wx[:, 0]), color='black')


# plt.xlabel(r'Posterior Mean $\left\langle z \right\rangle$', fontsize=14)
# plt.ylabel(r'$p(\left\langle z \right\rangle)$', fontsize=14)
# plt.legend(fontsize=10)
# plt.xlim([0.87, 1.3])
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.title('LSST SR, Photometric Likelihood, Post. Mean', fontsize=14)
# plt.savefig('lsst_sr_photometric_like_mean.pdf', bbox_inches='tight', dpi=300)

