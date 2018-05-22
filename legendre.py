import numpy as np
import argparse
from scipy.special import legendre


def legendre_coef(tpcf, l, delta_mu):
    """
    Calculate the coefficient of tpcf_l(s) in Legendre expansion for all s
    Inputs:
    + tpcf: ndarray
        the tpcf as a matrix dimensioned by mu and s
    + l: the order of desired Legendre polynomial
    Outputs:
    + coef:
        Return the coefficient of tpcf_l(s) as an array indexed by s values
    """
    if l % 2 == 1:
        raise ValueError("l has to be an even integer")
    temp = np.copy(tpcf)
    mu_vec = np.linspace(-1, 1, int(2/delta_mu))[:, None]  # the vector of mu values
    p = legendre(l)  # Legendre polynomial at order l
    mu_vec = np.array([p(mu) for mu in mu_vec])  # P-_l(mu)
    temp = temp * mu_vec * delta_mu  # P_l(mu) * tpcf(s, mu) * delta_mu
    temp = temp * (2 * l + 1) / 2
    return temp.sum(axis=0)


def legendre_test(delta_mu):
    """
    test with fixed s, i.e., tpcf as a function of only mu
    """
    num = int(2 / delta_mu)
    test1 = np.linspace(-1, 1, num)[:, None]  # tpcf(mu) = mu
    test2 = np.array([abs(mu) ** 1.5 for mu in test1])  # tpcf(mu) = |mu|^(1.5)
    test3 = np.array([1 / 2 * (3 * mu ** 2 - 1) for mu in test1])  # P_2
    test4 = np.array([1 / 8 * (35 * mu ** 4 - 30 * mu ** 2 + 3) for mu in test1])  # P_4
    test5 = np.array([1 / 16 * (231 * mu ** 6 - 315 * mu ** 4 + 105 * mu ** 2 - 5) for mu in test1])  # P_6
    for i in range(0, 20, 2):
        print(legendre_coef(test2, i, delta_mu))
    print()
    for i in range(0, 20, 2):
        print(legendre_coef(test3, i, delta_mu))
    print()
    for i in range(0, 20, 2):
        print(legendre_coef(test4, i, delta_mu))
    print()
    for i in range(0, 20, 2):
        print(legendre_coef(test5, i, delta_mu))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('delta_mu', type=float)
    args = parser.parse_args()
    legendre_test(args.delta_mu)
