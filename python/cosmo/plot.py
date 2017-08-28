""" Test plotting """


import sys
import numpy
import matplotlib
import matplotlib.pyplot

matplotlib.rc('font', size=15)


def main():
    """ Main """
    fname = sys.argv[1]
    with numpy.load(fname) as file:
        rand_rand = file["RR"]
        data_rand = file["DR"]
        data_data = file["DD"]
        correlation = file["TPCF"]
        correlation_ss = file["TPCFSS"]
        bins_s = file["BINS"]

    # create figure and subplots
    figure, axes = matplotlib.pyplot.subplots(2, 3, figsize=(15, 8))

    # plot RR(s)
    axes[0, 0].hist(bins_s[:-1], bins=bins_s, weights=rand_rand[0],
                    histtype="step", label="Weighted")
    axes[0, 0].hist(bins_s[:-1], bins=bins_s, weights=rand_rand[1],
                    histtype="step", label="Unweighted")
    # plot DR(s)
    axes[0, 1].hist(bins_s[:-1], bins=bins_s, weights=data_rand[0],
                    histtype="step", label="Weighted")
    axes[0, 1].hist(bins_s[:-1], bins=bins_s, weights=data_rand[1],
                    histtype="step", label="Unweighted")
    # plot DD(s)
    axes[0, 2].hist(bins_s[:-1], bins=bins_s, weights=data_data[0],
                    histtype="step", label="Weighted")
    axes[0, 2].hist(bins_s[:-1], bins=bins_s, weights=data_data[1],
                    histtype="step", label="Unweighted")
    # plot correlation
    axes[1, 0].hist(bins_s[:-1], bins=bins_s, weights=correlation[0],
                    histtype="step", label = "Weighted")
    axes[1, 0].hist(bins_s[:-1], bins=bins_s, weights=correlation[1],
                    histtype="step", label = "Unweighted")
    # plot correlation zoom-in
    axes[1, 1].hist(bins_s[:-1], bins=bins_s, weights=correlation[0],
                    histtype="step", label = "Weighted")
    axes[1, 1].hist(bins_s[:-1], bins=bins_s, weights=correlation[1],
                    histtype="step", label = "Unweighted")
    # plot correlation s^2
    axes[1, 2].hist(bins_s[:-1], bins=bins_s, weights=correlation_ss[0],
                    histtype="step", label = "Weighted")
    axes[1, 2].hist(bins_s[:-1], bins=bins_s, weights=correlation_ss[1],
                    histtype="step", label = "Unweighted")

    # set plot labels and legends
    leg_loc = ["upper left"]*3+["best"]*3
    y_label = ["RR(s)", "DR(s)", "DD(s)"] + [r"$\xi(s)$"]*2 + [r"$\xi(s)s^{2}$"]
    for i, axis in enumerate(axes.flat):
        axis.set(xlabel="s [Mpc/h]",
                 ylabel=y_label[i],
                 xlim= (0, 200))
        axis.legend(loc=leg_loc[i], fontsize=12)
    # zoom in
    axes[1, 1].set(xlim=[50, 200], ylim=[-0.0005, 0.015])

    figure.tight_layout()
    matplotlib.pyplot.show()


if __name__ == "__main__":
    main()