"""
python implementation of Hilbert Schmidt Independence Criterion
hsic_gam implements the HSIC test using a Gamma approximation
Python 2.7.12

Gretton, A., Fukumizu, K., Teo, C. H., Song, L., Scholkopf, B.,
& Smola, A. J. (2007). A kernel statistical test of independence.
In Advances in neural information processing systems (pp. 585-592).

Shoubo (shoubo.sub AT gmail.com)
09/11/2016

Inputs:
X 		n by dim_x matrix
Y 		n by dim_y matrix
alph 		level of test

Outputs:
testStat	test statistics
thresh		test threshold for level alpha test
"""
from copy import deepcopy
import numpy as np
from scipy.stats import gamma


def rbf_dot(pattern1, pattern2, deg):
    size1 = pattern1.shape
    size2 = pattern2.shape

    G = np.sum(pattern1 * pattern1, 1).reshape(size1[0], 1)
    H = np.sum(pattern2 * pattern2, 1).reshape(size2[0], 1)

    Q = np.tile(G, (1, size2[0]))
    R = np.tile(H.T, (size1[0], 1))

    H = Q + R - 2 * np.dot(pattern1, pattern2.T)

    H = np.exp(-H / 2 / (deg ** 2))

    return H


def hsic_gam(X, Y, alph=0.02):
    """
	X, Y are numpy vectors with row - sample, col - dim
	alph is the significance level
	auto choose median to be the kernel width
	:param X: 
	:param Y: 
	:param alph: 
	:return: 
	"""

    n = X.shape[0]

    Xmed = deepcopy(X)

    G = np.sum(Xmed * Xmed, 1).reshape(n, 1)
    Q = np.tile(G, (1, n))
    R = np.tile(G.T, (n, 1))

    dists = Q + R - 2 * np.dot(Xmed, Xmed.T)
    dists = np.tril(dists)
    # dists = dists.reshape(n ** 2, 1)

    width_x = np.sqrt(0.5 * np.median(dists[dists > 0]))  # 只是选出了距离的中位数
    # ----- -----

    # ----- width of X -----
    Ymed = deepcopy(Y)

    G = np.sum(Ymed * Ymed, 1).reshape(n, 1)
    Q = np.tile(G, (1, n))
    R = np.tile(G.T, (n, 1))

    dists = Q + R - 2 * np.dot(Ymed, Ymed.T)
    dists = dists - np.tril(dists)
    dists = dists.reshape(n ** 2, 1)

    width_y = np.sqrt(0.5 * np.median(dists[dists > 0]))
    # ----- -----

    bone = np.ones((n, 1), dtype=float)
    H = np.identity(n) - np.ones((n, n), dtype=float) / n  # H

    K = rbf_dot(X, X, width_x)  # This is Kxx
    L = rbf_dot(Y, Y, width_y)  # Lyy

    Kc = np.dot(np.dot(H, K), H)  # 矩阵乘法
    Lc = np.dot(np.dot(H, L), H)

    testStat = np.sum(Kc.T * Lc) / n

    varHSIC = (Kc * Lc / 6) ** 2  # why 6

    varHSIC = (np.sum(varHSIC) - np.trace(varHSIC)) / n / (n - 1)  #

    varHSIC = varHSIC * 72 * (n - 4) * (n - 5) / n / (n - 1) / (n - 2) / (n - 3)  # why 72

    K = K - np.diag(np.diag(K))  # clear diag..
    L = L - np.diag(np.diag(L))

    muX = np.dot(np.dot(bone.T, K), bone) / n / (n - 1)  #
    muY = np.dot(np.dot(bone.T, L), bone) / n / (n - 1)  #

    mHSIC = (1 + muX * muY - muX - muY) / n  #

    al = mHSIC ** 2 / varHSIC
    bet = varHSIC * n / mHSIC

    pval = 1 - gamma.cdf(testStat, al, scale=bet)[0][0]
    thresh = gamma.ppf(1 - alph, al, scale=bet)[0][0]

    """
    testStat < thresh : x and y independent
    else : not independent
    """
    # return testStat, thresh, "independent" if testStat < thresh else "not independent"
    # print(testStat, thresh)
    return True if testStat < thresh else False  # it means "independent"



if __name__ == '__main__':
    X = np.array([[0.1, 0.1, .2, 0.1, .2, .3]]).T
    Y = np.array([[0.2, 0.2, 0.2, 0.1, .1, .1]]).T
    print(X.shape)
    print(Y.shape)
    result = hsic_gam(X, Y)
    print(result)
