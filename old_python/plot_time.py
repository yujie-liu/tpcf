""" Plot time """

import sys
import numpy
import matplotlib
import matplotlib.pyplot

matplotlib.rc('font', size=15)


def main():
    """ Main function """
    # filename
    indir = 'out/timer'
    binw_s = numpy.array([5.88, 5.41, 5.00, 4.44, 4.00, 3.45, 2.99, 2.50, 2.00])
    fname_list = []
    for binw in binw_s:
        fname_list.append("{0}/time_{1:02.2f}kpc.txt".format(indir, binw))

    # open file and export data
    time = numpy.zeros((binw_s.size, 3, 3))
    for i, fname in enumerate(fname_list):
        time[i] = numpy.genfromtxt(fname)
    time = time*100
    time_min = numpy.min(time, axis=1)
    log_time_min = numpy.log(time_min)/numpy.log(10)

    # fit log of time
    coeff = numpy.polyfit(binw_s, log_time_min, 1)
    print(coeff[0])

    # plot
    figure, axes = matplotlib.pyplot.subplots(1, 2, figsize=(13, 5))

    # plot minimum time
    axes[0].plot(binw_s, time_min[:, 0], 'o-', label=r'RR(s)')
    axes[0].plot(binw_s, time_min[:, 1], 'o-', label=r'DR(s)')
    axes[0].plot(binw_s, time_min[:, 2], 'o-', label=r'DD(s)')
    axes[0].set_ylabel(r'Run Time [s]')

    # plot log of minimum time
    axes[1].plot(binw_s, log_time_min[:, 0], 'o-', label=r'RR(s)')
    axes[1].plot(binw_s, log_time_min[:, 1], 'o-', label=r'DR(s)')
    axes[1].plot(binw_s, log_time_min[:, 2], 'o-', label=r'DD(s)')
    axes[1].set_ylabel(r'Log Run Time')

    for axis in axes:
        axis.set(xlabel='Binwidth [Mpc/h]')
        axis.legend()
        axis.grid(linestyle='--')

    figure.tight_layout()
    matplotlib.pyplot.show()
    # matplotlib.pyplot.savefig('plot/time.png')


if __name__ == "__main__":
    main()