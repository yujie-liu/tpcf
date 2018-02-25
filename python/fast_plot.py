""" Plotting for NPZ output """

import argparse

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc('font', size=15)

def main():
    """ Main """
    print('Fast plotting modules')

    parser = argparse.ArgumentParser(
        description='Fast plotting for two-point correlation function.')
    parser.add_argument('filenames', type=str, nargs='+',
                        help='Path to two-point correlation function.')
    parser.add_argument('-s', '-S', '--save', type=str, default=None,
                        help='Path to save plot output. Disable showing plot.')
    parser.add_argument('-e', '-E', '--error', action='store_true', default=False)
    parser.add_argument('--version', action='version', version='KITCAT 1.10')
    args = parser.parse_args()

    # Create figure and subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Read in file
    for i, fname in enumerate(args.filenames):
        with np.load(fname) as f:
            rand_rand = f['rr']
            data_rand = f['dr']
            data_data = f['dd']
            tpcf = f['tpcf']
            tpcfss = f['tpcfss']
            bins = f['bins']

        # Plot
        label = '{}'.format(i+1)
        if not args.error:
            axes[0, 0].hist(bins[:-1], bins=bins, weights=rand_rand[0], histtype='step', label=label)
            axes[0, 1].hist(bins[:-1], bins=bins, weights=data_rand[0], histtype='step', label=label)
            axes[0, 2].hist(bins[:-1], bins=bins, weights=data_data[0], histtype='step', label=label)
            axes[1, 0].hist(bins[:-1], bins=bins, weights=tpcf[0], histtype="step", label=label)
            axes[1, 1].hist(bins[:-1], bins=bins, weights=tpcfss[0], histtype="step", label=label)
        else:
            s =  (bins[:-1]+bins[1:])/2
            axes[0, 0].errorbar(s, rand_rand[0], yerr=rand_rand[1], label=label, fmt='--.')
            axes[0, 1].errorbar(s, data_rand[0], yerr=data_rand[1], label=label, fmt='--.')
            axes[0, 2].errorbar(s, data_data[0], yerr=data_data[1], label=label, fmt='--.')
            axes[1, 0].errorbar(s, tpcf[0], yerr=tpcf[1], label=label, fmt='--.')
            axes[1, 1].errorbar(s, tpcfss[0], yerr=tpcfss[1], label=label, fmt='--.')
        axes[1, 2].axis('off')


    # set plot labels and legends
    y_label = ["RR(s)", "DR(s)", "DD(s)"] + [r"$\xi(s)$"] + [r"$\xi(s)s^{2}$"]
    for i, ax in enumerate(axes.flat[:-1]):
        ax.set(xlabel="s [Mpc/h]", ylabel=y_label[i], xlim=(0, 200))
        if len(args.filenames) > 1:
            if i in np.arange(0, 3):
                ax.legend(loc='upper left')
            else:
                ax.legend()

    fig.tight_layout()

    # Save plot or show plot
    if args.save is not None:
        plt.savefig('{}'.format(args.save), bbox_inches='tight')
    else:
        plt.show()


if __name__ == "__main__":
    main()
