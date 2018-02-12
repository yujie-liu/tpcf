""" Plotting for NPZ output """


import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc('font', size=15)

def main():
    """ Main """
    fname_list = sys.argv[1:]

    # Create figure and subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Read in file
    for fname in fname_list:
        with np.load(fname) as f:
            rand_rand = f['rr']
            data_rand = f['dr']
            data_data = f['dd']
            tpcf = f['tpcf']
            tpcfss = f['tpcfss']
            bins = f['bins']

        # Plot
        axes[0, 0].hist(bins[:-1], bins=bins, weights=rand_rand[0], histtype='step')
        axes[0, 1].hist(bins[:-1], bins=bins, weights=data_rand[0], histtype='step')
        axes[0, 2].hist(bins[:-1], bins=bins, weights=data_data[0], histtype='step')
        axes[1, 0].hist(bins[:-1], bins=bins, weights=tpcf[0], histtype="step")
        axes[1, 1].hist(bins[:-1], bins=bins, weights=tpcf[0], histtype="step")
        axes[1, 1].set(xlim=[50, 200], ylim=[-0.005, 0.005])
        axes[1, 2].hist(bins[:-1], bins=bins, weights=tpcfss[0], histtype="step")

    # set plot labels and legends
    y_label = ["RR(s)", "DR(s)", "DD(s)"] + [r"$\xi(s)$"]*2 + [r"$\xi(s)s^{2}$"]
    for i, ax in enumerate(axes.flat):
        ax.set(xlabel="s [Mpc/h]", ylabel=y_label[i], xlim=(0, 200))

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()