''' Main to run correlation_function.'''

import numpy
import matplotlib
import matplotlib.pyplot
from correlation_function import CorrelationFunction

matplotlib.rc('font', size=15)

def main():
    ''' Main '''
    tpcf = CorrelationFunction('config.cfg')

    # Calculating normalized factor
    norm = numpy.array([tpcf.normalization(weighted=True),
                        tpcf.normalization(weighted=False)])
    print("Normalization Factor:")
    print("+ Weighted", norm[0])
    print("+ Unweighted", norm[1])

    # Construct RR(s)
    rand_rand, bins_s = tpcf.rand_rand()
    rand_rand[0] = rand_rand[0]/norm[0][0]
    rand_rand[1] = rand_rand[1]/norm[1][0]

    # Construct DR(s)
    data_rand, bins_s = tpcf.data_rand()
    data_rand[0] = data_rand[0]/norm[0][1]
    data_rand[1] = data_rand[1]/norm[1][1]

    # Construct DD(s)
    data_data, bins_s = tpcf.data_data()
    data_data[0] = data_data[0]/norm[0][2]
    data_data[1] = data_data[1]/norm[1][2]

    # Construct two-point correlation function
    correlation = [tpcf.correlation(rand_rand[i], data_rand[i], data_data[i],
                                    bins_s) for i in range(2)]

    # Save histogram
    numpy.savez("out/tpcf_North", RR=rand_rand, DR=data_rand, DD=data_data,
                TPCF=correlation[0], TPCFSS=correlation[1],
                BINS=bins_s)

    # Save plot
    figure, axes = matplotlib.pyplot.subplots(2, 2, figsize=(12, 8), sharex=True)
    centers_s = 0.5*(bins_s[:-1]+bins_s[1:])
    axes[0, 0].plot(centers_s, data_data[0], label='DD(s)')
    axes[0, 0].plot(centers_s, data_rand[0], label='DR(s)')
    axes[0, 0].plot(centers_s, rand_rand[0], label='RR(s)')
    axes[0, 1].plot(centers_s, data_data[1], label='DD(s)')
    axes[0, 1].plot(centers_s, data_rand[1], label='DR(s)')
    axes[0, 1].plot(centers_s, rand_rand[1], label='DD(s)')
    axes[1, 0].plot(centers_s, correlation[0][0], label='Weighted')
    axes[1, 0].plot(centers_s, correlation[1][0], label='Unweighted')
    axes[1, 1].plot(centers_s, correlation[0][1], label='Weighted')
    axes[1, 1].plot(centers_s, correlation[1][1], label='Unweighted')

    axes[0, 0].set(ylabel='Normed Counts',
                   title='Weighted')
    axes[0, 1].set(ylabel='Normed Counts',
                   title='Unweighted')
    axes[1, 0].set(ylabel=r'$\xi(s)$')
    axes[1, 1].set(ylabel=r'$\xi(s)s^{2}$')
    for ax in axes.flat:
        ax.set(xlabel='s [Mpc/h]')
        ax.legend()

    figure.tight_layout()


    matplotlib.pyplot.savefig("North.png", bbox_inches='tight')


if __name__ == "__main__":
    main()
