1/0
bias_ratio_reconstructed = model_bias.get_bias(data_wx[:, 0])
print(bias_ratio_reconstructed  - data_wx[:, 5])
nb = data_wx[:, 1]*data_wx[:, -1]/data_wx[:, -3]/data_wx[:, 3]
nb_rec = data_wx[:, 1]*data_wx[:, -1]/bias_ratio_reconstructed/data_wx[:, 3]

samples_wrb = multivariate_normal.rvs(data_wx[:, 1], np.diag((data_wx[:, 2]/4.)**2), size=10000)
samples_rr = multivariate_normal.rvs(data_wx[:, 3], np.diag((data_wx[:, 4]/4.)**2), size=10000)





print(model_bias.get_model_low())
model_bias.set_model_low(model_bias.get_model_low()+0.1)

bias_ratio_reconstructed = model_bias.get_bias(data_wx[:, 0])
nb_rec_list = []
for i in range(len(samples_rr)):
    nb_rec_list.append(samples_wrb[i]*data_wx[:, -1]/bias_ratio_reconstructed/samples_rr[i])

nb_rec_list = np.array(nb_rec_list)



plt.plot(data_wx[:, 0], data_wx[:, -2])
plt.plot(data_wx[:, 0], deconvolution_res['res'][-1]/np.trapz(deconvolution_res['res'][-1],
         data_wx[:, 0]))
plt.plot(data_wx[:, 0], nb_rec)
plt.fill_between(data_wx[:, 0], np.percentile(nb_rec_list, 10, axis=0), np.percentile(nb_rec_list, 90, axis=0),
                 alpha=0.2)
plt.show()


