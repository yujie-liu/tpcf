""" Use timeit to measure run time of divide.py """


import sys
import timeit
import numpy
from correlation_function import CorrelationFunction


def main():
    """ Main """

    config_flist = sys.argv[1:]

    for fname in config_flist:
        # set up correlation function
        print(fname)
        tpcf = CorrelationFunction(fname)

        # output filename
        bins_s = tpcf._CorrelationFunction__bins_s
        binw_s = bins_s[1]-bins_s[0]
        output_fname = 'out/timer/time_{0:2.2f}kpc.txt'.format(binw_s)
        print(output_fname)

        # Set up timer
        rr_timer = timeit.Timer(lambda: tpcf.angular_distance(0, 100))
        dr_timer = timeit.Timer(lambda: tpcf.r_angular_distance(0, 100))
        dd_timer = timeit.Timer(lambda: tpcf.pairs_separation(0, 100, out='DD'))
        rr_run_time = rr_timer.repeat(repeat=3, number=1)
        dr_run_time = dr_timer.repeat(repeat=3, number=1)
        dd_run_time = dd_timer.repeat(repeat=3, number=1)

        # Print out output
        print("Min, Max:")
        print("- RR: [{0:.4f} sec, {1:.4f}] sec".format(min(rr_run_time),
                                                        max(rr_run_time)))
        print("- DR: [{0:.4f} sec, {1:.4f}] sec".format(min(dr_run_time),
                                                        max(dr_run_time)))
        print("- DD: [{0:.4f} sec, {1:.4f}] sec".format(min(dd_run_time),
                                                        max(dd_run_time)))

        # Save output
        run_time = numpy.array([rr_run_time, dr_run_time, dd_run_time]).T
        header = "RR (s) DR(s) DD(s)"
        numpy.savetxt(output_fname, run_time, header=header)


if __name__ == "__main__":
    main()
