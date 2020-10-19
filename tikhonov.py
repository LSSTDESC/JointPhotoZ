import numpy as np



def tikhonov(hnew, regul=0.1):
    bmatrix = np.diag(np.ones((len(hnew),)))
    result_inverse = np.zeros((len(hnew), len(hnew)))

    for i in range(len(hnew)):
        if isinstance(regul, np.float):
            tikmat = regul * np.copy(bmatrix)
        else:
            tikmat = regul[i] * np.copy(bmatrix)
        b = bmatrix[:, i]
        result_inverse[:, i] = np.linalg.inv(hnew.T.dot(hnew) + (tikmat.T).dot(tikmat)).dot((hnew.T).dot(b))

    return result_inverse

