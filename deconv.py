import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.stats import dirichlet
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.stats import cauchy
from scipy.stats import lognorm
from scipy.special import erf
import pickle
import sys

def transformation_logit(pi_vec):
    return np.log(pi_vec[:-1]/pi_vec[-1] + sys.float_info.epsilon)

def backtransform_logit(s_vec):
    denom = 1. + np.sum(np.exp(s_vec))

    vec_pi = np.exp(s_vec)/denom
    vec_pi = vec_pi.tolist()
    vec_pi.append(1/denom)
    return np.array(vec_pi)


def factory_derivative(pi_vec, pi_trafo):
    sum_pi_trafo = (1. + np.sum([np.exp(el) for el in pi_trafo]))
    def get_derivative(i, j):
        if (i == j) and (i < len(pi_vec) - 1):
            return pi_vec[i] - pi_vec[i]**2
        elif (i != j) and (i < len(pi_vec) - 1):
            return -pi_vec[i]*pi_vec[j]
        else:
            return -pi_vec[j]*1./sum_pi_trafo


    def get_secondderiv(alpha, i, j):
        if (i == j) and (i < len(pi_vec) - 1):
            deriv_ialpha = get_derivative(i, alpha)
            return deriv_ialpha - 2.*pi_vec[i]*deriv_ialpha
        elif (i != j) and (i < len(pi_vec) - 1):
            deriv_ialpha = get_derivative(i, alpha)
            deriv_jalpha = get_derivative(j, alpha)
            return -deriv_ialpha*pi_vec[j] - pi_vec[i]*deriv_jalpha
        else:
            deriv_jalpha = get_derivative(j, alpha)
            return (pi_vec[j]*pi_vec[alpha] - deriv_jalpha)*1./sum_pi_trafo

    return get_derivative, get_secondderiv



class BenchmarkHistogram(object):
    def __init__(self, samples, grid):
        self.grid = grid
        self.midpoints = self.grid[:-1] + (self.grid[1:] - self.grid[:-1])/2.
        self.delta_z = (self.grid[1:] - self.grid[:-1])/2.
        self.samples = samples


class BenchmarkNoisy(object):
    def __init__(self, benchmark_hist):
        print('initialize BenchmarkNoisy')
        self.bench_hist = benchmark_hist
        self.samples_noisy = np.copy(self.bench_hist.samples)
        # graham_model = np.loadtxt('ErrorModelGraham.dat')
        # graham_model = np.loadtxt('ErrorModelGraham.dat')
        # idx_sorted = graham_model[:, 0].argsort()

        # data_graham = np.column_stack((graham_model[idx_sorted[9:], 0], graham_model[idx_sorted[9:], 1]))
        # data_graham = np.row_stack((np.array([0.2, 0.0598]), data_graham))
        # model_graham = InterpolatedUnivariateSpline(data_graham[:, 0], data_graham[:, 1], ext=3, k=2)
        # from matplotlib import pyplot as plt

        # plt.plot(np.linspace(0.0, 2.5, num=100), 0.02*(1. + np.linspace(0.0, 2.5, num=100)))
        # plt.plot(np.linspace(0.0, 2.5, num=100), model_graham(np.linspace(0.0, 2.5, num=100)))
        # plt.show()
        # 1/0

        self.std = 0.02 * (1. + self.bench_hist.samples)
        self.samples_noisy = np.random.normal(self.samples_noisy, scale=self.std)

        self.grid = self.bench_hist.grid

        self.grid_like = np.zeros((len(self.samples_noisy),
                                   len(self.grid) - 1))
        print('discretize pdf')

        for j in range(len(self.grid) - 1):
            #each grid point
            self.grid_like[:, j] = norm.cdf(self.grid[j+1], loc=self.samples_noisy, scale=self.std) \
                                 - norm.cdf(self.grid[j], loc=self.samples_noisy, scale=self.std)


    def get_ml(self, n_iter, ini_vec):
        #np.random.seed(0)
        print('get_ml')
        phi = np.ones(len(self.grid)-1)
        ini_vec = ini_vec/np.float(np.sum(ini_vec))
        print(np.sum(ini_vec))
        assert np.isclose(np.sum(ini_vec), 1)
        assert np.all(ini_vec >= 0)
        pz_list = [ini_vec]
        for nel in range(n_iter):
            print(nel)
            list_nk = []
            denominator = pz_list[-1].dot(self.grid_like.T)
            nominator = pz_list[-1]*self.grid_like

            list_nk = nominator / denominator[:, None]

            nk = np.sum(list_nk, axis=0)
            pz_list.append(nk/np.sum(nk))
        return pz_list


def get_hessian_y(grid_like, max_values, pi_trafo):
    deriv, secondderiv = factory_derivative(max_values, pi_trafo)
    hessian = np.zeros((len(pi_trafo), len(pi_trafo)))

    Ntot = grid_like.shape[1]

    deriv_mat = np.zeros((Ntot, len(hessian)))
    for dj in range(deriv_mat.shape[0]):
        for di in range(deriv_mat.shape[1]):
            deriv_mat[dj, di] = deriv(dj, di)

    second_deriv_mat = np.zeros((Ntot, len(pi_trafo), len(pi_trafo)))
    for j in range(second_deriv_mat.shape[0]):
        for alpha in range(second_deriv_mat.shape[1]):
            for z in range(second_deriv_mat.shape[2]):
                second_deriv_mat[j, alpha, z] = secondderiv(alpha, j, z)


    print('done')

    denominator1 = grid_like.dot(max_values)
    for z in range(len(hessian)):
        nominator1_first = deriv_mat[:, z].dot(grid_like.T)
        for alpha in range(len(hessian)):
            nominator1_second = deriv_mat[:, alpha].dot(grid_like.T)
            #second term
            nominator2 = second_deriv_mat[:, alpha, z].dot(grid_like.T)
            first_term = - (nominator1_first * nominator1_second)/(denominator1**2)
            second_term = nominator2/denominator1
            curr_sum_terms = first_term + second_term
            hessian[z, alpha] = np.sum(curr_sum_terms)

    return hessian



def run_model(data_z, midpoints, iter, ini_model):

    delta_z = (midpoints[1] - midpoints[0])/2.
    breaks = midpoints - delta_z
    breaks = breaks.tolist()
    breaks.append(breaks[-1] + 2.*delta_z)
    breaks = np.array(breaks)

    bench_hist = BenchmarkHistogram(data_subsample, breaks)
    bench_noisy = BenchmarkNoisy(bench_hist)
    res = bench_noisy.get_ml(iter, ini_model(midpoints))
    pi_trafo = transformation_logit(res[-1])
    deriv, secondderiv = factory_derivative(res[-1], pi_trafo)
    h_new = get_hessian_y(bench_noisy.grid_like, res[-1], pi_trafo)

    return res, pi_trafo, h_new



if __name__ == '__main__':
    import json
    import pandas as pd
    data_subsample = pd.read_hdf('DES_DNF_WX_binning_zspec.fits', key='zspec').sample(50000)
    data_subsample = np.array(np.array(data_subsample)).flatten()
    wx_data = np.loadtxt('new_cross_correlation_measurements_analytic_bias0.01t0.10Mpc.dat')
    fname_out = 'LSST_IDEAL_DESI_25bins_cut_high_50000_good_ini.json'


    # graham_model = np.loadtxt('ErrorModelGraham.dat')
    # graham_model = np.loadtxt('ErrorModelGraham.dat')
    # idx_sorted = graham_model[:, 0].argsort()

    # data_graham = np.column_stack((graham_model[idx_sorted[9:], 0], graham_model[idx_sorted[9:], 1]))
    # data_graham = np.row_stack((np.array([0.2, 0.0598]), data_graham))
    # model_graham = InterpolatedUnivariateSpline(data_graham[:, 0], data_graham[:, 1], ext=3, k=2)
    # # from matplotlib import pyplot as plt

    # # plt.plot(np.linspace(0.0, 2.5, num=100), 0.02*(1. + np.linspace(0.0, 2.5, num=100)))
    # # plt.plot(np.linspace(0.0, 2.5, num=100), model_graham(np.linspace(0.0, 2.5, num=100))*(1 + np.linspace(0.0, 2.5, num=100)))
    # # plt.show()
    # # 1/0

    midpoints = np.linspace(0.0, np.max(wx_data[:, 0]), num=25)
    hist, histgrid = np.histogram(data_subsample, 50, normed=True)
    mids_hist = histgrid[:-1] + (histgrid[1] - histgrid[0])/2.0
    hist = hist/np.sum(hist)
    model = InterpolatedUnivariateSpline(mids_hist, hist, k=1, ext=1)

    res, pi_trafo, h_new = run_model(data_subsample, midpoints, iter=400, ini_model=model)


    h_new_list = [el.tolist() for el in h_new]
    res_new_list = [el.tolist() for el in res]
    midpoints_list = midpoints.tolist()
    pi_trafo_list = pi_trafo.tolist()

    output = {'res': res_new_list, 'pi_trafo': pi_trafo_list, 'h_new': h_new_list, 'midpoints': midpoints_list, 'sample_size': len(data_subsample)}

    with open(fname_out, 'w') as outfile:
        json.dump(output, outfile)






