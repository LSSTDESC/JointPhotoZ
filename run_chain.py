import numpy as np
# from matplotlib import pyplot as plt
import pickle
from scipy.stats import gaussian_kde
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import Rbf
# from sklearn.decomposition import PCA
from scipy.stats import norm, multivariate_normal
from like_wx import *
from deconv import *
from like_tf import *



def run_chain(fname_deconv):
    ##############################################
    #READ IN THE WX DATA AND INITIALIZE THE MODEL#
    ##############################################
    data_wx = np.loadtxt('new_cross_correlation_measurements_analytic_bias.dat')


    with open('output_model_new_cross_correlation_measurements_analytic_bias.pickle', 'rb') as infile:
        output_model = pickle.load(infile)


    model_bias = GeneratorBiasModel(output_model, data_wx[:, 0])

    wxcov_mod = 4
    model_loglike_wx = LogLikeWX(output_model, data_wx, wxcov_mod)

    #######################################################
    # READ IN THE TF DECONVOLUITON AND DEFINE THE LOGLIKE #
    #######################################################

    import json
    with open(fname_deconv, 'r') as infile:
        deconvolution_res = pickle.load(infile)

    len_model = len(deconvolution_res['mean_res'])

    midpoints = data_wx[:, 0]
    model_tf = ListLikeTF(deconvolution_res, midpoints)



    ######################################
    # Definition of the Joint Likelihood #
    ######################################

    trace_vec_s = [np.array(deconvolution_res['mean_res'][0])]

    trace_bias = [np.array([model_bias.get_model_low()[0],
                  model_bias.get_model_mid()[0],
                  model_bias.get_model_high()[0],
                  model_bias.get_model_low()[1],
                  model_bias.get_model_mid()[1],
                  model_bias.get_model_high()[1],
                  model_bias.get_model_low()[2],
                  model_bias.get_model_mid()[2],
                  model_bias.get_model_high()[2]])]



    trace_loglike = []

    # list_step_nz = np.sqrt(np.diag(-np.linalg.inv(deconvolution_res['hess_traf'])))
    def propose_nz(idx, vec_s):
        list_step_nz = 5*np.array([0.02302523, 0.00672454, 0.00620927,
                                 0.01084058, 0.01117412, 0.01068708,
                                 0.0107255,  0.01219014, 0.02442341,
                                 0.02102579, 0.01899059, 0.02605574,
                                 0.05406305, 0.3852213, 0.3649834,
                                 0.3510255, 0.3012916, 0.3370887,
                                 0.22215714])

        nz_s_proposed = np.random.normal(vec_s[idx], list_step_nz[idx])
        res_vec_s = np.copy(vec_s)
        res_vec_s[idx] = nz_s_proposed
        return res_vec_s


    def propose_b(idx, vec_b):
        list_step_bias = 0.7*np.array([0.008, 0.02, 0.05, 0.05, 0.01, 0.02, 0.01, 0.01, 0.02])
        bias_proposed = np.random.normal(vec_b[idx], list_step_bias[idx])
        res_vec_b = np.copy(vec_b)
        res_vec_b[idx] = bias_proposed
        return res_vec_b

    def loglike(nz_s, vec_b):
        pi_curr = backtransform_logit(nz_s)
        return model_tf.loglike(nz_s) + model_loglike_wx.log(pi_curr, vec_b)

    def propose_idx_model():
        return np.random.randint(len_model)



    n_samp = 500000
    len_vec_s = len(trace_vec_s[0])
    print(len_vec_s)

    len_vec_b = len(trace_bias[-1])

    for idx_samp in range(n_samp):
        if idx_samp%100 == 0:
            print(idx_samp)
            trace_vec_s_out = np.array(trace_vec_s)
            trace_bias_out = np.array(trace_bias)
            trace_loglike_out = np.array(trace_loglike)

            np.savetxt(X=trace_vec_s_out, fname=fname_deconv + str(wxcov_mod) + 'mod_chain1_trace_vec_s_test_large_std_bratio_second_DESI_version2_largechain.dat')
            np.savetxt(X=trace_bias_out, fname=fname_deconv + str(wxcov_mod) + 'mod_chain1_trace_bias_test_large_std_bratio_second_DESI_version2_largechain.dat')
            np.savetxt(X=trace_loglike_out, fname=fname_deconv + str(wxcov_mod) + 'mod_chain1_trace_loglike_bratio_second_DESI_version2_largechain.dat')

        vec_s = np.copy(trace_vec_s[-1])
        for nz_idx in range(len_vec_s):
            proposed_nz = propose_nz(nz_idx, vec_s)

            log_new = loglike(proposed_nz, trace_bias[-1])
            log_old = loglike(vec_s, trace_bias[-1])
            log_accept_ratio = log_new - log_old
            rnd_curr = np.log(np.random.uniform(low=0.0, high=1.0))
            if log_accept_ratio > rnd_curr:
                vec_s = proposed_nz
                trace_loglike.append(log_new)
            else:
                trace_loglike.append(log_old)

        trace_vec_s.append(vec_s)

        vec_b = np.copy(trace_bias[-1])
        for b_idx in range(len_vec_b):
            proposed_b = propose_b(b_idx, vec_b)

            log_new = loglike(trace_vec_s[-1], proposed_b)
            log_old = loglike(trace_vec_s[-1], vec_b)
            log_accept_ratio = log_new - log_old
            rnd_curr = np.log(np.random.uniform(low=0.0, high=1.0))
            if log_accept_ratio > rnd_curr:
                vec_b = proposed_b
                trace_loglike.append(log_new)
            else:
                trace_loglike.append(log_old)

        trace_bias.append(vec_b)

        # propose_model = propose_idx_model()
        # print(propose_model)
        # log_new = loglike(trace_vec_s[-1], trace_bias[-1], propose_model)
        # log_old = loglike(trace_vec_s[-1], trace_bias[-1], trace_idx_model[-1])
        # log_accept_ratio = log_new - log_old
        # print(log_accept_ratio)
        # rnd_curr = np.log(np.random.uniform(low=0.0, high=1.0))
        # if log_accept_ratio > rnd_curr:
        #     print('accept')
        #     trace_idx_model.append(propose_model)
        # else:
        #     trace_idx_model.append(trace_idx_model[-1])







list_dec = ['results_new_final/output_regularization_models_31_bins_500k_0_08.pickle']

for el in list_dec:
    print(el)
    run_chain(el)
