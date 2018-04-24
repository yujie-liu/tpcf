""" Apply Fourier transform to calculate and plot the power spectrum from tpcf """

import argparse

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import trapz

from lib.myio import load

mpl.rc('font', size=15)
mpl.rc('figure', figsize=(8, 5))

def power_spectrum(k, xi, r):
    """ Numerical integration for power spectrum """
    xi = xi.reshape(-1, 1)
    r = r.reshape(-1, 1)
    k = k.reshape(1, -1)
    integrand = xi*np.sinc(k*r)*r**2
    return np.squeeze(2*np.pi*trapz(integrand, r, axis=0))

def main():
    """ Main """
    print('Power Spectrum Module')
    parser = argparse.ArgumentParser(
        description='Plot the power spectrum from the two-point correlation function.')
    parser.add_argument('filenames', type=str, help='Path to two-point correlation function.')
    parser.add_argument('--min', type=float, default=0, help='k Min.')
    parser.add_argument('--max', type=float, default=10, help='k Max.')
    parser.add_argument('--step', type=float, default=0.01, help='k Step.')
    parser.add_argument('-w', '-W', '--weighted', action='store_true', default=False)
    parser.add_argument('-o', '-O', '--output', type=str, default=None,
                        help='Path to save plot output. Disable showing plot.')
    parser.add_argument('-e', '-E', '--error', action='store_true', default=False)
    parser.add_argument('--version', action='version', version='KITCAT 1.10')
    args = parser.parse_args()

    # Create figure
    fig, ax = plt.subplots(1)

    # Read in file
    correlations = next(load(args.filenames))
    for i, correlation in enumerate(correlations):
        # Get RR, DR, DD, tpcf and tpcfss
        tpcf = correlation.tpcf(args.weighted)
        bins = correlation.bins
        bins = (bins[:-1]+bins[0])/2

        # Calculate and plot the power spectrum
        k = np.arange(args.min, args.max, args.step)
        ps = power_spectrum(k, tpcf[0], bins)

        label = '{}'.format(i+1)
        ax.plot(k, ps, '-.', label=label)

    # set plot labels and legends
    ax.set(xlabel='k', ylabel='P(k)', xscale='log')
    ax.legend()

    fig.tight_layout()

    # Save plot or show plot
    if args.output is not None:
        plt.savefig('{}'.format(args.output), bbox_inches='tight')
    else:
        plt.show()


if __name__ == "__main__":
    main()