import numpy as np
import pickle
from matplotlib import pyplot as plt
from tikhonov import *
import json
from scipy.stats import multivariate_normal
from scipy.interpolate import InterpolatedUnivariateSpline


def transformation_logit(pi_vec):
    return np.log(pi_vec[:-1]/(pi_vec[-1]+np.finfo(float).tiny) + np.finfo(float).tiny)

def backtransform_logit(s_vec):
    denom = 1. + np.sum(np.exp(s_vec))

    vec_pi = np.exp(s_vec)/denom
    vec_pi = vec_pi.tolist()
    vec_pi.append(1/denom)
    return np.array(vec_pi)



with open('LSST_IDEAL_DESI_50bins_cut_high_5000000_good_ini.json', 'r') as infile:
    data70 = json.load(infile)


def merge_neighboring_bins(midpoints, data_pi, num=35):
    midpoints_split = np.array_split(midpoints, num)
    midpoints_new = np.array([np.mean(el) for el in midpoints_split])

    data_pi_vec = []
    for el in data_pi:
        data_pi_vec.append([np.mean(el_new) for el_new in np.array_split(el, num)])

    data_pi_vec = np.array(data_pi_vec)
    return midpoints_new, data_pi_vec



import json
import pandas as pd
data_subsample = pd.read_hdf('DES_DNF_WX_binning_zspec.fits', key='zspec')

pi_trafo = np.array(data70['pi_trafo'])
midpoints = np.array(data70['midpoints'])

delta_z = midpoints[1] - midpoints[0]
breaks_new = np.copy(midpoints) - delta_z
breaks_new = breaks_new.tolist()
breaks_new.append(breaks_new[-1] + delta_z)
breaks_new = np.array(breaks_new)
hist_true = np.histogram(data_subsample, breaks_new)

def get_merged_hist(num):
    merged_true = merge_neighboring_bins(midpoints, [hist_true[0]/np.float(np.sum(hist_true[0]))], num=num)

    size_hnew = len(np.array(data70['h_new']))

    tikvec = np.ones(size_hnew) * 10

    cov_trafo = -tikhonov(np.array(data70['h_new']), tikvec)  #0.001

    data_s = multivariate_normal.rvs(pi_trafo, cov_trafo, size=10000)
    data_pi = np.array([backtransform_logit(el) for el in data_s])

    merged_bins = merge_neighboring_bins(midpoints, data_pi, num=num)


    normed_merged_bins = np.array([el/np.trapz(el, merged_bins[0]) for el in merged_bins[1]])
    mean_merged_bins = np.array([np.trapz(el*merged_bins[0], merged_bins[0]) for el in normed_merged_bins])

    return merged_true, normed_merged_bins, mean_merged_bins


def estimate_gaussian_cov(element, new_midpoints):
    list_output = []
    for el in element[1]:
        model = InterpolatedUnivariateSpline(element[0][0], el, k=1, ext=0)
        res = model(new_midpoints)
        res = res/np.sum(res)
        pi_trafo = transformation_logit(res)
        if np.sum([np.isnan(el) for el in pi_trafo]) > 0:
            print('nan detected')
            continue
        else:
            list_output.append(pi_trafo)

    list_output = np.array(list_output)
    mean_gauss = np.mean(list_output, axis=0)

    cov_gauss = np.cov(list_output.T)

    return mean_gauss, cov_gauss


result_merged_20 = get_merged_hist(25)
bin_list = [25]  #np.arange(20, 22, 1)
result_list_merged = []
for el in bin_list:
    print(el)
    new_merged_hist = get_merged_hist(el)

    output = {'merged_true': new_merged_hist[0],
              'normed_merged_bins': new_merged_hist[1],
              'mean_merged_bins': new_merged_hist[2]}

    # with open('/Users/markusmichaelrau/Desktop/new_measurements_ideal/setups/Graham_DESI/50bins/plots/final_result_plot/result_wo_prior.pickle',
    #           'wb') as outfile:
    #     pickle.dump(output, outfile)
    result_list_merged.append(new_merged_hist)


data_wx = np.loadtxt('new_cross_correlation_measurements_analytic_bias.dat')


list_mean_res = []
list_cov_res = []

for el in result_list_merged:
    mean_res, cov_res = estimate_gaussian_cov(el, data_wx[:, 0])
    mean_res = mean_res.tolist()
    cov_res = [el.tolist() for el in cov_res]
    list_mean_res.append(mean_res)
    list_cov_res.append(cov_res)

import pickle
output = {'cov_res': list_cov_res, 'mean_res': list_mean_res,
          'result_list_merged': result_list_merged}
with open('results_new_final/output_regularization_models_50_25_bins_5000k_10.pickle', 'w') as outfile:
    pickle.dump(output, outfile)


for el in result_list_merged:
    plt.hist(el[2], 1000)
    print(np.std(el[2]))

    # plt.fill_between(el[0][0], np.percentile(el[1], 16, axis=0),
    #     np.percentile(el[1], 84, axis=0), alpha=0.5)


print(np.trapz(data_wx[:, -2]*data_wx[:, 0], data_wx[:, 0]) - np.mean(np.array(data_subsample)))
plt.axvline(np.mean(np.array(data_subsample)), color='black')
plt.axvline(np.trapz(data_wx[:, -2]*data_wx[:, 0], data_wx[:, 0]), color='red')
plt.axvline(np.trapz(result_list_merged[0][0][1][0]/np.trapz(result_list_merged[0][0][1][0],
    result_list_merged[0][0][0])*result_list_merged[0][0][0],  result_list_merged[0][0][0]), color='blue')
# print(result_list_merged[0][0][1])
print(result_list_merged[0][0][0])

# plt.plot(result_list_merged[0][0][0], result_list_merged[0][0][1][0]/np.trapz(result_list_merged[0][0][1][0],
#          result_list_merged[0][0][0]), '.', color='grey', alpha=0.8)
# plt.plot(data_wx[:, 0], data_wx[:, -2], '--', color='black')

plt.title('LSST SR, N=50k, T=0.04, AB=31')
plt.show()
